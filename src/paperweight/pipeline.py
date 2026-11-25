"""Orchestrates the four compiler stages and persists artifacts between them."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paperweight.config import Settings
from paperweight.devin import DevinClient
from paperweight.github import GitHubClient
from paperweight.ingest import load_paper
from paperweight.llm import LLMClient, make_llm
from paperweight.models import Paper, SessionResult, Spec
from paperweight.spec import generate_spec, render_spec_markdown

STAGES = ("parse", "spec", "implement", "test")
SPEC_DIR = "docs/paperweight"
IMPLEMENT_BRANCH = "paperweight/implementation"
TEST_BRANCH = "paperweight/verification"
SPEC_SOURCE = "spec.paper.sha256"

IMPLEMENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["pr_url", "branch"],
    "properties": {
        "pr_url": {"type": "string"},
        "branch": {"type": "string"},
        "modules": {"type": "array", "items": {"type": "string"}},
        "edge_cases_handled": {"type": "array", "items": {"type": "string"}},
        "deviations_from_spec": {"type": "array", "items": {"type": "string"}},
    },
}

TEST_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["pr_url", "branch"],
    "properties": {
        "pr_url": {"type": "string"},
        "branch": {"type": "string"},
        "coverage_percent": {"type": "number"},
        "invariants_covered": {"type": "array", "items": {"type": "string"}},
        "invariants_violated": {"type": "array", "items": {"type": "string"}},
        "benchmarks": {"type": "array", "items": {"type": "string"}},
    },
}


def _paper_digest(paper: Paper) -> str:
    return hashlib.sha256(paper.model_dump_json().encode()).hexdigest()


class PipelineError(RuntimeError):
    pass


Logger = Callable[[str], None]


@dataclass
class PipelineOptions:
    source: str | None
    out_dir: Path
    repo: str | None = None
    private: bool = True
    from_stage: str = "parse"
    until_stage: str = "test"
    dry_run: bool = False
    wait: bool = True
    poll_interval: float = 30.0
    session_timeout: float = 4 * 3600
    max_acu_limit: int | None = None
    tags: list[str] = field(default_factory=lambda: ["paperweight"])


@dataclass
class PipelineResult:
    paper: Paper | None = None
    spec: Spec | None = None
    sessions: list[SessionResult] = field(default_factory=list)

    @property
    def pr_urls(self) -> list[str]:
        return [url for s in self.sessions for url in s.pr_urls]


def _stage_range(start: str, stop: str) -> list[str]:
    for name in (start, stop):
        if name not in STAGES:
            raise PipelineError(f"unknown stage {name!r}; expected one of {', '.join(STAGES)}")
    i, j = STAGES.index(start), STAGES.index(stop)
    if i > j:
        raise PipelineError(f"--from {start} comes after --until {stop}")
    return list(STAGES[i : j + 1])


def split_repo(repo: str) -> tuple[str, str]:
    owner, sep, name = repo.partition("/")
    if not sep or not owner or not name or "/" in name:
        raise PipelineError(f"--repo must look like OWNER/NAME, got {repo!r}")
    return owner, name


def implement_prompt(repo: str, default_branch: str, spec: Spec, paper: Paper) -> str:
    title = paper.metadata.title or spec.project_name
    return f"""You are the implementation stage of paperweight, an arXiv-to-production compiler.

Paper: {title} {paper.metadata.source_url or ""}
Target repository: https://github.com/{repo} (default branch `{default_branch}`)

The repository already contains the engineering spec produced from the paper:
- `{SPEC_DIR}/SPEC.md` (human-readable)
- `{SPEC_DIR}/spec.json` (typed schemas, components, invariants, edge cases, test plan)
- `{SPEC_DIR}/paper.json` (extracted equations, algorithms, figures)

Authentication: the `GITHUB_TOKEN` secret has push and PR access to the target repository.
Clone with `git clone https://x-access-token:$GITHUB_TOKEN@github.com/{repo}.git` and use
`GH_TOKEN=$GITHUB_TOKEN gh ...` for pull requests.

Do the following:
1. Create branch `{IMPLEMENT_BRANCH}` from `{default_branch}`.
2. Scaffold a production Python package named `{spec.project_name}`: `pyproject.toml` (uv,
   Python >= 3.11), `src/` layout, ruff config, a README explaining the algorithm and linking
   the paper, and a GitHub Actions workflow running ruff and pytest.
3. Implement every component and function in spec.json with the exact signatures and types,
   faithful to the paper's equations and pseudocode. Use numpy where it is the natural fit.
4. Isolate numerical/algorithmic edge cases listed in the spec (and any others you find) in
   explicit, documented validation and handling code; raise typed exceptions for invalid input.
5. Add a small smoke test per public function so CI is green. The verification stage will
   add the full property-based suite afterwards, so do not write it here.
6. Run ruff and pytest locally, then open a pull request from `{IMPLEMENT_BRANCH}` into
   `{default_branch}` titled "Implement {spec.project_name} from paper". The description must
   map each spec component to its module and list any deviations from the spec.
7. Report the PR URL, branch, modules, edge cases handled and deviations in structured output.
Do not merge the pull request.
"""


def test_prompt(repo: str, base_branch: str, spec: Spec, paper: Paper) -> str:
    title = paper.metadata.title or spec.project_name
    return f"""You are the verification stage of paperweight, an arXiv-to-production compiler.

Paper: {title} {paper.metadata.source_url or ""}
Target repository: https://github.com/{repo}
Implementation branch: `{base_branch}`
Spec: `{SPEC_DIR}/SPEC.md` and `{SPEC_DIR}/spec.json` on that branch.

Authentication: the `GITHUB_TOKEN` secret has push and PR access to the target repository.
Clone with `git clone https://x-access-token:$GITHUB_TOKEN@github.com/{repo}.git` and use
`GH_TOKEN=$GITHUB_TOKEN gh ...` for pull requests.

Build a test and verification suite derived from the paper's mathematical guarantees:
1. Create branch `{TEST_BRANCH}` from `{base_branch}`.
2. Property-based tests (Hypothesis): one or more tests for every invariant in spec.json,
   named after the invariant id (e.g. `test_inv_1_...`), with input strategies that respect
   the preconditions and deliberately probe boundaries.
3. Unit tests for every `test_plan` entry of kind "unit" and every edge case in the spec.
4. Synthetic-input benchmarks (pytest-benchmark) for every complexity bound, generating inputs
   at several sizes and asserting the empirical growth is consistent with the stated bound.
5. Configure coverage (pytest-cov) and add it to CI. Aim for >= 90% line coverage of the
   package. If a test reveals the implementation violates a paper guarantee, fix the
   implementation in this branch and call it out in the PR description.
6. Run the suite, then open a pull request from `{TEST_BRANCH}` into `{base_branch}` titled
   "Add verification suite for {spec.project_name}". Include a table mapping each invariant
   to its tests and the final coverage number.
7. Report the PR URL, branch, coverage, invariants covered/violated and benchmarks in
   structured output.
Do not merge the pull request.
"""


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        options: PipelineOptions,
        *,
        llm: LLMClient | None = None,
        devin: DevinClient | None = None,
        github: GitHubClient | None = None,
        log: Logger = print,
    ):
        self.settings = settings
        self.options = options
        self._llm = llm
        self._devin = devin
        self._github = github
        self.log = log
        self.out = options.out_dir

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = make_llm(self.settings)
        return self._llm

    @property
    def devin(self) -> DevinClient:
        if self._devin is None:
            self.settings.require("devin_api_key", "devin_org_id")
            assert self.settings.devin_api_key and self.settings.devin_org_id
            self._devin = DevinClient(
                self.settings.devin_api_key,
                self.settings.devin_org_id,
                self.settings.devin_api_base,
            )
        return self._devin

    @property
    def github(self) -> GitHubClient:
        if self._github is None:
            self.settings.require("github_token")
            assert self.settings.github_token
            self._github = GitHubClient(self.settings.github_token, self.settings.github_api_base)
        return self._github

    def _write(self, name: str, content: str) -> Path:
        path = self.out / name
        path.write_text(content)
        return path

    def _read_model(self, name: str, model: type[Any]) -> Any:
        path = self.out / name
        if not path.exists():
            raise PipelineError(f"{path} not found; run the earlier stage first")
        return model.model_validate_json(path.read_text())

    def run(self) -> PipelineResult:
        opts = self.options
        stages = _stage_range(opts.from_stage, opts.until_stage)
        needs_devin = any(s in stages for s in ("implement", "test")) and not opts.dry_run
        if needs_devin:
            if not opts.repo:
                raise PipelineError("--repo OWNER/NAME is required for the implement/test stages")
            split_repo(opts.repo)
            self.settings.require("devin_api_key", "devin_org_id", "github_token")
        self.out.mkdir(parents=True, exist_ok=True)
        result = PipelineResult()

        if "parse" in stages:
            if not opts.source:
                raise PipelineError("a paper source (arXiv URL/id or file) is required")
            self.log(f"[1/4] ingest & parse: {opts.source}")
            result.paper = load_paper(opts.source)
            self._write("paper.json", result.paper.model_dump_json(indent=2))
            p = result.paper
            self.log(
                f"      {p.metadata.title!r}: {len(p.equations)} equations, "
                f"{len(p.algorithms)} algorithms, {len(p.figures)} figures"
            )
        else:
            result.paper = self._read_model("paper.json", Paper)

        if "spec" in stages:
            self.log("[2/4] formal specification")
            result.spec = generate_spec(result.paper, self.llm)
            self._write("spec.json", result.spec.model_dump_json(indent=2))
            self._write(SPEC_SOURCE, _paper_digest(result.paper))
            self._write("SPEC.md", render_spec_markdown(result.spec, result.paper))
            s = result.spec
            self.log(
                f"      {s.project_name}: {len(s.components)} components, "
                f"{len(s.invariants)} invariants, {len(s.test_plan)} planned tests"
            )
        elif any(st in stages for st in ("implement", "test")):
            result.spec = self._read_model("spec.json", Spec)
            source = self.out / SPEC_SOURCE
            if source.exists() and source.read_text().strip() != _paper_digest(result.paper):
                raise PipelineError(
                    "spec.json was generated from a different paper.json; "
                    "rerun with --from spec to regenerate it"
                )

        if opts.dry_run:
            if result.spec and any(st in stages for st in ("implement", "test")):
                repo = opts.repo or f"OWNER/{result.spec.project_name}"
                self._write(
                    "implement_prompt.md", implement_prompt(repo, "main", result.spec, result.paper)
                )
                self._write(
                    "test_prompt.md", test_prompt(repo, IMPLEMENT_BRANCH, result.spec, result.paper)
                )
                self.log(f"dry run: Devin prompts written to {self.out}")
            self._write_report(result)
            return result

        if not any(st in stages for st in ("implement", "test")):
            self._write_report(result)
            return result

        assert result.spec is not None
        if "implement" in stages:
            self._record(result, self._implement(result.paper, result.spec))
        if "test" in stages:
            if "implement" in stages and not opts.wait:
                self.log(
                    "[4/4] skipped: --no-wait leaves no implementation branch to test yet; "
                    "rerun with --from test once the implementation PR is open"
                )
            else:
                base = self._implementation_branch(result)
                self._record(result, self._verify(result.paper, result.spec, base))

        self._write_report(result)
        return result

    def _record(self, result: PipelineResult, session: SessionResult) -> None:
        result.sessions.append(session)
        if self.options.wait and not session.pr_urls:
            self._write_report(result)
            raise PipelineError(
                f"{session.stage} session {session.session_url} finished without a PR "
                f"(status {session.status}/{session.status_detail})"
            )

    def _seed_repo(self, paper: Paper, spec: Spec) -> str:
        assert self.options.repo
        owner, name = split_repo(self.options.repo)
        repo = self.github.ensure_repo(
            owner, name, private=self.options.private, description=spec.summary
        )
        branch = repo.get("default_branch") or "main"
        files = {
            f"{SPEC_DIR}/SPEC.md": render_spec_markdown(spec, paper),
            f"{SPEC_DIR}/spec.json": spec.model_dump_json(indent=2) + "\n",
            f"{SPEC_DIR}/paper.json": paper.model_dump_json(indent=2) + "\n",
        }
        for path, content in files.items():
            self.github.put_file(owner, name, path, content, f"paperweight: add {path}", branch)
        self.log(f"      seeded https://github.com/{owner}/{name} ({branch}) with the spec")
        return branch

    def _run_session(
        self, stage: str, prompt: str, schema: dict[str, Any], title: str
    ) -> SessionResult:
        assert self.settings.github_token
        session = self.devin.create_session(
            prompt,
            title=title,
            tags=[*self.options.tags, f"paperweight:{stage}"],
            structured_output_schema=schema,
            session_secrets={"GITHUB_TOKEN": self.settings.github_token},
            max_acu_limit=self.options.max_acu_limit,
        )
        session_id = session["session_id"]
        self.log(f"      Devin session: {session.get('url', session_id)}")
        self._write(f"{stage}_session.json", json.dumps(session, indent=2))
        if self.options.wait:
            session = self.devin.wait(
                session_id,
                poll_interval=self.options.poll_interval,
                timeout=self.options.session_timeout,
                on_update=lambda s: self.log(
                    f"      status: {s.get('status')} / {s.get('status_detail')}"
                ),
            )
            self._write(f"{stage}_session.json", json.dumps(session, indent=2))
        output = session.get("structured_output") or None
        pr_urls = [pr["pr_url"] for pr in session.get("pull_requests") or [] if pr.get("pr_url")]
        if output and output.get("pr_url") and output["pr_url"] not in pr_urls:
            pr_urls.insert(0, output["pr_url"])
        result = SessionResult(
            stage=stage,
            session_id=session_id,
            session_url=session.get("url", ""),
            status=session.get("status", "unknown"),
            status_detail=session.get("status_detail"),
            pr_urls=pr_urls,
            branch=(output or {}).get("branch"),
            structured_output=output,
        )
        for url in pr_urls:
            self.log(f"      PR: {url}")
        return result

    def _implement(self, paper: Paper, spec: Spec) -> SessionResult:
        self.log("[3/4] scaffolding & implementation")
        default_branch = self._seed_repo(paper, spec)
        assert self.options.repo
        return self._run_session(
            "implement",
            implement_prompt(self.options.repo, default_branch, spec, paper),
            IMPLEMENT_OUTPUT_SCHEMA,
            f"paperweight: implement {spec.project_name}",
        )

    def _implementation_branch(self, result: PipelineResult) -> str:
        for session in result.sessions:
            if session.stage == "implement":
                return session.branch or IMPLEMENT_BRANCH
        path = self.out / "implement_session.json"
        if path.exists():
            output = json.loads(path.read_text()).get("structured_output") or {}
            return output.get("branch") or IMPLEMENT_BRANCH
        return IMPLEMENT_BRANCH

    def _verify(self, paper: Paper, spec: Spec, base_branch: str) -> SessionResult:
        self.log("[4/4] test & verification suite")
        assert self.options.repo
        return self._run_session(
            "test",
            test_prompt(self.options.repo, base_branch, spec, paper),
            TEST_OUTPUT_SCHEMA,
            f"paperweight: verify {spec.project_name}",
        )

    def _write_report(self, result: PipelineResult) -> None:
        lines = ["# paperweight report", ""]
        if result.paper:
            m = result.paper.metadata
            lines += [
                f"- Paper: {m.title} {m.source_url or ''}".rstrip(),
                f"- Extracted: {len(result.paper.equations)} equations, "
                f"{len(result.paper.algorithms)} algorithms, {len(result.paper.figures)} figures",
            ]
        if result.spec:
            s = result.spec
            lines.append(
                f"- Spec: `{s.project_name}` with {len(s.components)} components, "
                f"{len(s.invariants)} invariants, {len(s.edge_cases)} edge cases"
            )
        if self.options.repo:
            lines.append(f"- Repository: https://github.com/{self.options.repo}")
        for session in result.sessions:
            lines.append(
                f"- {session.stage}: {session.session_url} "
                f"({session.status}/{session.status_detail})"
            )
            lines += [f"  - PR: {url}" for url in session.pr_urls]
        self._write("report.md", "\n".join(lines) + "\n")

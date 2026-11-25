from __future__ import annotations

import json

import pytest

from paperweight.config import ConfigError, Settings
from paperweight.pipeline import (
    IMPLEMENT_BRANCH,
    Pipeline,
    PipelineError,
    PipelineOptions,
    split_repo,
)

SETTINGS = Settings(devin_api_key="k", devin_org_id="org-1", github_token="ghtok")


class FakeGitHub:
    def __init__(self):
        self.files: dict[str, str] = {}
        self.created: list[tuple[str, str, bool]] = []

    def ensure_repo(self, owner, name, *, private=True, description=""):
        self.created.append((owner, name, private))
        return {"default_branch": "main"}

    def put_file(self, owner, name, path, content, message, branch):
        self.files[path] = content


class FakeDevin:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts: list[dict] = []
        self.sessions: dict[str, dict] = {}

    def create_session(self, prompt, **kwargs):
        sid = f"devin-{len(self.prompts) + 1}"
        self.prompts.append({"prompt": prompt, **kwargs})
        self.sessions[sid] = self.outputs.pop(0)
        return {"session_id": sid, "url": f"https://app.devin.ai/sessions/{sid}"}

    def wait(self, session_id, **kwargs):
        kwargs["on_update"]({"status": "running", "status_detail": "finished"})
        return {
            "session_id": session_id,
            "url": f"https://app.devin.ai/sessions/{session_id}",
            "status": "running",
            "status_detail": "finished",
            **self.sessions[session_id],
        }


def _options(tmp_path, fixture_dir, **kw):
    return PipelineOptions(source=str(fixture_dir), out_dir=tmp_path, **kw)


def test_full_pipeline(tmp_path, fixture_dir, spec_json, fake_llm_factory):
    github = FakeGitHub()
    devin = FakeDevin(
        [
            {
                "structured_output": {"pr_url": "https://github.com/me/out/pull/1", "branch": "b1"},
                "pull_requests": [{"pr_url": "https://github.com/me/out/pull/1"}],
            },
            {"structured_output": {"pr_url": "https://github.com/me/out/pull/2", "branch": "b2"}},
        ]
    )
    logs: list[str] = []
    result = Pipeline(
        SETTINGS,
        _options(tmp_path, fixture_dir, repo="me/out"),
        llm=fake_llm_factory([json.dumps(spec_json)]),
        devin=devin,
        github=github,
        log=logs.append,
    ).run()

    assert result.pr_urls == [
        "https://github.com/me/out/pull/1",
        "https://github.com/me/out/pull/2",
    ]
    assert github.created == [("me", "out", True)]
    assert set(github.files) == {
        "docs/paperweight/SPEC.md",
        "docs/paperweight/spec.json",
        "docs/paperweight/paper.json",
    }
    implement, verify = devin.prompts
    assert implement["session_secrets"] == {"GITHUB_TOKEN": "ghtok"}
    assert "paperweight:implement" in implement["tags"]
    assert "fast_smoothing" in implement["prompt"]
    assert "from `b1`" in verify["prompt"]
    for name in ("paper.json", "spec.json", "SPEC.md", "report.md", "test_session.json"):
        assert (tmp_path / name).exists()
    assert "pull/2" in (tmp_path / "report.md").read_text()


def test_tests_not_started_without_implementation_pr(
    tmp_path, fixture_dir, spec_json, fake_llm_factory
):
    devin = FakeDevin([{"structured_output": None}])
    pipeline = Pipeline(
        SETTINGS,
        _options(tmp_path, fixture_dir, repo="me/out"),
        llm=fake_llm_factory([json.dumps(spec_json)]),
        devin=devin,
        github=FakeGitHub(),
        log=lambda _: None,
    )
    with pytest.raises(PipelineError, match="without a PR"):
        pipeline.run()
    assert len(devin.prompts) == 1


def test_dry_run_writes_prompts(tmp_path, fixture_dir, spec_json, fake_llm_factory):
    Pipeline(
        Settings(),
        _options(tmp_path, fixture_dir, dry_run=True),
        llm=fake_llm_factory([json.dumps(spec_json)]),
        log=lambda _: None,
    ).run()
    assert "OWNER/fast_smoothing" in (tmp_path / "implement_prompt.md").read_text()
    assert IMPLEMENT_BRANCH in (tmp_path / "test_prompt.md").read_text()


def test_resume_from_test_stage(tmp_path, fixture_dir, spec_json, fake_llm_factory):
    Pipeline(
        Settings(),
        _options(tmp_path, fixture_dir, until_stage="spec"),
        llm=fake_llm_factory([json.dumps(spec_json)]),
        log=lambda _: None,
    ).run()
    (tmp_path / "implement_session.json").write_text(
        json.dumps({"structured_output": {"branch": "impl-x"}})
    )
    devin = FakeDevin([{"structured_output": {"pr_url": "p", "branch": "t"}}])
    Pipeline(
        SETTINGS,
        PipelineOptions(source=None, out_dir=tmp_path, repo="me/out", from_stage="test"),
        devin=devin,
        github=FakeGitHub(),
        log=lambda _: None,
    ).run()
    assert "from `impl-x`" in devin.prompts[0]["prompt"]


def test_requires_credentials_for_devin_stages(tmp_path, fixture_dir):
    with pytest.raises(ConfigError, match="DEVIN_API_KEY"):
        Pipeline(Settings(), _options(tmp_path, fixture_dir, repo="me/out")).run()


def test_requires_repo(tmp_path, fixture_dir):
    with pytest.raises(PipelineError, match="--repo"):
        Pipeline(SETTINGS, _options(tmp_path, fixture_dir)).run()


@pytest.mark.parametrize("bad", ["noslash", "/x", "a/", "a/b/c"])
def test_split_repo_rejects(bad):
    with pytest.raises(PipelineError):
        split_repo(bad)


def test_stage_order_validated(tmp_path, fixture_dir):
    with pytest.raises(PipelineError, match="comes after"):
        Pipeline(
            SETTINGS, _options(tmp_path, fixture_dir, from_stage="test", until_stage="spec")
        ).run()


def test_parse_only_needs_no_credentials(tmp_path, fixture_dir):
    result = Pipeline(
        Settings(), _options(tmp_path, fixture_dir, until_stage="parse"), log=lambda _: None
    ).run()
    assert result.paper is not None and result.spec is None
    assert (tmp_path / "paper.json").exists() and (tmp_path / "report.md").exists()

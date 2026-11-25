# paperweight

An autonomous arXiv-to-production compiler: give it a paper, get back a GitHub repository
with an engineering spec, an implementation and a verification suite, delivered as pull requests.

```
arXiv URL / PDF
  └─[1] ingest & parse ─────────── paper.json  (equations, algorithms, figures, theorems)
      └─[2] formal spec (LLM) ──── spec.json + SPEC.md  (typed schemas, invariants, bounds, edge cases, test plan)
          └─[3] implement (Devin) ─ PR #1: package scaffold + core algorithms + edge-case handling
              └─[4] verify (Devin) ─ PR #2: Hypothesis property tests, unit tests, benchmarks, coverage
```

1. **Ingest & parse.** Fetches arXiv metadata and the LaTeX source (falls back to the PDF when
   no source is available or the source only wraps a PDF). Extracts equations, `algorithm` blocks,
   figures/TikZ diagrams, theorems and section text, resolving `\input`/`\include`.
2. **Formal spec.** An LLM (Anthropic or OpenAI) turns the paper into a `Spec`
   ([`models.py`](src/paperweight/models.py)): data schemas, components with typed signatures and
   pseudocode, invariants traceable to the paper, complexity bounds, edge cases and a test plan.
   Replies are schema-validated and retried on error.
3. **Scaffold & implement.** Creates the output repo (if needed), commits the spec to
   `docs/paperweight/`, and starts a Devin session that implements the package on
   `paperweight/implementation` and opens a PR.
4. **Test & verify.** A second Devin session branches from the implementation, writes property
   tests for every invariant, unit tests for every edge case, benchmarks for every complexity
   bound, and opens a PR into the implementation branch.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in, and export, the keys below
```

| Variable | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` | stage 2 (`PAPERWEIGHT_LLM_PROVIDER` / `PAPERWEIGHT_LLM_MODEL` optional) |
| `DEVIN_API_KEY`, `DEVIN_ORG_ID` | stages 3–4 (service user or PAT with `UseDevinSessions`) |
| `GITHUB_TOKEN` | stages 3–4: creates the repo, seeds the spec; passed to Devin as a session secret to push and open PRs |

## Usage

```bash
# full pipeline
uv run paperweight compile https://arxiv.org/abs/1412.6980 --repo my-org/adam-optimizer

# individual stages
uv run paperweight parse 1706.03762 -o out/          # paper.json
uv run paperweight spec paper.pdf -o out/            # + spec.json, SPEC.md
uv run paperweight compile 1706.03762 --dry-run      # + Devin prompts, no API calls to Devin/GitHub

# resume from saved artifacts in out/
uv run paperweight compile --from implement --repo my-org/adam-optimizer
uv run paperweight compile --from test --repo my-org/adam-optimizer
```

Useful flags: `--public`, `--no-wait`, `--max-acu N`, `--poll-interval`, `--session-timeout`.
Artifacts (`paper.json`, `spec.json`, `SPEC.md`, `*_session.json`, `report.md`) are written to
`--out` (default `out/`); PR URLs are printed to stdout.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

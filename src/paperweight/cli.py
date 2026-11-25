from __future__ import annotations

import sys
from pathlib import Path

import typer

from paperweight import __version__
from paperweight.config import ConfigError, Settings
from paperweight.devin import DevinError
from paperweight.github import GitHubError
from paperweight.ingest import IngestError
from paperweight.llm import LLMError
from paperweight.pipeline import STAGES, Pipeline, PipelineError, PipelineOptions

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="paperweight: compile arXiv papers into production repositories.",
)

KNOWN_ERRORS = (ConfigError, DevinError, GitHubError, IngestError, LLMError, PipelineError)


def _log(message: str) -> None:
    typer.echo(message, err=True)


def _run(settings: Settings, options: PipelineOptions) -> None:
    try:
        result = Pipeline(settings, options, log=_log).run()
    except KNOWN_ERRORS as exc:
        _log(f"error: {exc}")
        raise typer.Exit(1) from exc
    _log(f"artifacts: {options.out_dir}")
    for url in result.pr_urls:
        typer.echo(url)


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Show version and exit."
    ),
) -> None:
    pass


@app.command()
def compile(
    source: str = typer.Argument(None, help="arXiv URL/id, PDF, .tex, source tarball or dir."),
    repo: str = typer.Option(None, "--repo", "-r", help="Output GitHub repository OWNER/NAME."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Artifact directory."),
    public: bool = typer.Option(False, "--public", help="Create the output repo as public."),
    from_stage: str = typer.Option("parse", "--from", help=f"First stage: {', '.join(STAGES)}."),
    until_stage: str = typer.Option("test", "--until", help="Last stage to run."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Stop before calling Devin/GitHub; write prompts instead."
    ),
    no_wait: bool = typer.Option(
        False, "--no-wait", help="Do not wait for Devin; run tests later with --from test."
    ),
    poll_interval: float = typer.Option(30.0, help="Seconds between Devin status polls."),
    session_timeout: float = typer.Option(4 * 3600, help="Max seconds to wait per session."),
    max_acu: int = typer.Option(None, help="ACU limit per Devin session."),
) -> None:
    """Run the full pipeline: parse -> spec -> implement -> test."""
    _run(
        Settings.from_env(),
        PipelineOptions(
            source=source,
            out_dir=out,
            repo=repo,
            private=not public,
            from_stage=from_stage,
            until_stage=until_stage,
            dry_run=dry_run,
            wait=not no_wait,
            poll_interval=poll_interval,
            session_timeout=session_timeout,
            max_acu_limit=max_acu,
        ),
    )


@app.command()
def parse(
    source: str = typer.Argument(..., help="arXiv URL/id, PDF, .tex, source tarball or dir."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Artifact directory."),
) -> None:
    """Stage 1 only: extract equations, algorithms and figures into paper.json."""
    _run(Settings.from_env(), PipelineOptions(source=source, out_dir=out, until_stage="parse"))


@app.command()
def spec(
    source: str = typer.Argument(..., help="arXiv URL/id, PDF, .tex, source tarball or dir."),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Artifact directory."),
) -> None:
    """Stages 1-2: parse the paper and write spec.json and SPEC.md."""
    _run(Settings.from_env(), PipelineOptions(source=source, out_dir=out, until_stage="spec"))


if __name__ == "__main__":
    sys.exit(app())

"""Stage 2: turn a parsed Paper into a typed engineering Spec."""

from __future__ import annotations

import json

from pydantic import ValidationError

from paperweight.llm import LLMClient, LLMError, extract_json
from paperweight.models import Paper, Spec

PROMPT_CHAR_BUDGET = 150_000

SYSTEM_PROMPT = """You are a staff engineer converting a research paper into a production \
engineering specification. Be precise and faithful to the paper: every invariant, bound and \
edge case must be traceable to an equation, algorithm, theorem or statement in the paper \
(cite it in `source`). Prefer concrete Python type hints. Invariants must be testable \
properties suitable for property-based testing. Include numerical edge cases (empty inputs, \
zero/negative values, overflow/underflow, degenerate shapes, NaN/inf) wherever relevant. \
Reply with a single JSON object that validates against this JSON Schema and nothing else:

{schema}"""


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n[...truncated...]"


def build_paper_digest(paper: Paper, budget: int = PROMPT_CHAR_BUDGET) -> str:
    meta = paper.metadata
    parts = [
        f"# {meta.title or 'Untitled paper'}",
        f"arXiv: {meta.arxiv_id}" if meta.arxiv_id else "",
        f"Authors: {', '.join(meta.authors)}" if meta.authors else "",
        f"## Abstract\n{meta.abstract}" if meta.abstract else "",
    ]
    if paper.equations:
        eqs = "\n".join(
            f"- [{e.label or e.environment}] (section: {e.section or '-'})\n  {e.latex}"
            for e in paper.equations
        )
        parts.append(f"## Equations\n{eqs}")
    if paper.algorithms:
        algos = "\n\n".join(
            f"### {a.caption or a.label or 'Algorithm'}\n{a.body}" for a in paper.algorithms
        )
        parts.append(f"## Algorithms\n{algos}")
    if paper.theorems:
        parts.append("## Theorems and definitions\n" + "\n".join(f"- {t}" for t in paper.theorems))
    if paper.figures:
        figs = "\n".join(f"- [{f.kind}] {f.caption}" for f in paper.figures)
        parts.append(f"## Figures and diagrams\n{figs}")
    head = "\n\n".join(p for p in parts if p)

    remaining = max(budget - len(head), 0)
    body_parts = []
    for section in paper.sections:
        if remaining <= 0:
            break
        chunk = _truncate(f"### {section.title}\n{section.text}", remaining)
        body_parts.append(chunk)
        remaining -= len(chunk)
    if body_parts:
        head += "\n\n## Paper text\n" + "\n\n".join(body_parts)
    return _truncate(head, budget)


def generate_spec(paper: Paper, llm: LLMClient, max_attempts: int = 3) -> Spec:
    system = SYSTEM_PROMPT.format(schema=json.dumps(Spec.model_json_schema()))
    user = build_paper_digest(paper)
    error = ""
    for _ in range(max_attempts):
        prompt = user
        if error:
            prompt += f"\n\nYour previous reply was invalid:\n{error}\nReturn corrected JSON only."
        reply = llm.complete(system, prompt)
        try:
            return Spec.model_validate(extract_json(reply))
        except (LLMError, ValidationError) as exc:
            error = str(exc)[:4000]
    raise LLMError(f"could not obtain a valid spec after {max_attempts} attempts: {error}")


def render_spec_markdown(spec: Spec, paper: Paper | None = None) -> str:
    lines = [f"# {spec.project_name} — engineering spec", ""]
    if paper and paper.metadata.title:
        source = paper.metadata.source_url or paper.metadata.arxiv_id or ""
        lines += [f"Derived from **{paper.metadata.title}** {source}".rstrip(), ""]
    lines += [spec.summary, ""]

    if spec.schemas:
        lines += ["## Data schemas", ""]
        for schema in spec.schemas:
            lines += [f"### `{schema.name}`", schema.description, ""]
            lines += ["| field | type | constraints | description |", "|---|---|---|---|"]
            for f in schema.fields:
                cons = "; ".join(f.constraints)
                lines.append(f"| `{f.name}` | `{f.type}` | {cons} | {f.description} |")
            lines.append("")

    if spec.components:
        lines += ["## Components", ""]
        for comp in spec.components:
            lines += [f"### {comp.name} (`{comp.module}`)", comp.responsibility, ""]
            for fn in comp.functions:
                lines += [f"#### `{fn.signature}`", fn.description, ""]
                if fn.preconditions:
                    lines += ["Preconditions:"] + [f"- {p}" for p in fn.preconditions] + [""]
                if fn.postconditions:
                    lines += ["Postconditions:"] + [f"- {p}" for p in fn.postconditions] + [""]
                if fn.invariants:
                    lines += [f"Invariants: {', '.join(fn.invariants)}", ""]
                if fn.complexity:
                    c = fn.complexity
                    lines += [f"Complexity: time {c.time or '?'}, space {c.space or '?'}", ""]
                if fn.pseudocode:
                    lines += ["```", fn.pseudocode.rstrip(), "```", ""]

    if spec.invariants:
        lines += ["## Invariants", "", "| id | statement | formal | source |", "|---|---|---|---|"]
        for inv in spec.invariants:
            formal = inv.formal.replace("|", "\\|")
            lines.append(f"| {inv.id} | {inv.statement} | `{formal}` | {inv.source} |")
        lines.append("")

    if spec.complexity:
        lines += ["## Complexity bounds", "", "| operation | time | space | source |"]
        lines.append("|---|---|---|---|")
        for c in spec.complexity:
            lines.append(f"| {c.operation} | {c.time} | {c.space} | {c.source} |")
        lines.append("")

    if spec.edge_cases:
        lines += ["## Edge cases", ""]
        for e in spec.edge_cases:
            lines.append(f"- **{e.id}** {e.description} → {e.expected_behavior}")
        lines.append("")

    if spec.test_plan:
        lines += ["## Test plan", "", "| id | kind | target | invariants | description |"]
        lines.append("|---|---|---|---|---|")
        for t in spec.test_plan:
            invs = ", ".join(t.invariants)
            lines.append(f"| {t.id} | {t.kind.value} | {t.target} | {invs} | {t.description} |")
        lines.append("")

    if spec.open_questions:
        lines += ["## Open questions", ""] + [f"- {q}" for q in spec.open_questions] + [""]
    return "\n".join(lines).rstrip() + "\n"

"""Extract equations, algorithms, figures and structure from LaTeX source."""

from __future__ import annotations

import posixpath
import re
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass

from paperweight.models import Algorithm, Equation, Figure, Paper, PaperMetadata, Section

EQUATION_ENVS = (
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "eqnarray",
    "eqnarray*",
    "displaymath",
    "flalign",
    "flalign*",
)
ALGORITHM_ENVS = ("algorithm", "algorithm*")
FIGURE_ENVS = ("figure", "figure*")
THEOREM_ENVS = (
    "theorem",
    "lemma",
    "proposition",
    "corollary",
    "definition",
    "claim",
    "assumption",
)
SECTION_LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3}
MAX_INPUT_DEPTH = 10

_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_INPUT_RE = re.compile(r"\\(?:input|include|subfile)\s*\{([^}]+)\}")
_SECTION_RE = re.compile(r"\\(section|subsection|subsubsection)\*?\s*(?:\[[^\]]*\])?\s*\{")
_BEGIN_RE = re.compile(r"\\begin\s*\{([^}]+)\}")
_GRAPHICS_RE = re.compile(r"\\includegraphics\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")


@dataclass(frozen=True)
class EnvMatch:
    name: str
    start: int
    end: int
    body: str


def strip_comments(tex: str) -> str:
    return "\n".join(_COMMENT_RE.sub("", line) for line in tex.splitlines())


def read_braced(tex: str, open_index: int) -> tuple[str, int]:
    """Return the content of the brace group opening at ``open_index`` and the index after it."""
    if open_index >= len(tex) or tex[open_index] != "{":
        raise ValueError("expected '{'")
    depth = 0
    i = open_index
    while i < len(tex):
        ch = tex[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return tex[open_index + 1 : i], i + 1
        i += 1
    return tex[open_index + 1 :], len(tex)


def command_argument(tex: str, command: str) -> str | None:
    match = re.search(r"\\" + re.escape(command) + r"\s*(?:\[[^\]]*\])?\s*\{", tex)
    if not match:
        return None
    content, _ = read_braced(tex, match.end() - 1)
    return content.strip()


def find_environments(tex: str, names: tuple[str, ...]) -> list[EnvMatch]:
    """Find top-level occurrences of the given environments, handling same-name nesting."""
    wanted = set(names)
    matches: list[EnvMatch] = []
    pos = 0
    while True:
        begin = _BEGIN_RE.search(tex, pos)
        if not begin:
            break
        name = begin.group(1).strip()
        if name not in wanted:
            pos = begin.end()
            continue
        token_re = re.compile(r"\\(begin|end)\s*\{" + re.escape(name) + r"\}")
        depth = 1
        scan = begin.end()
        end_match = None
        while depth:
            token = token_re.search(tex, scan)
            if not token:
                break
            depth += 1 if token.group(1) == "begin" else -1
            scan = token.end()
            end_match = token
        if depth or end_match is None:
            pos = begin.end()
            continue
        matches.append(
            EnvMatch(name, begin.start(), end_match.end(), tex[begin.end() : end_match.start()])
        )
        pos = end_match.end()
    return matches


def resolve_inputs(tex: str, files: Mapping[str, str], base_dir: str = "", depth: int = 0) -> str:
    """Inline ``\\input``/``\\include`` directives using the provided file map."""
    if depth > MAX_INPUT_DEPTH:
        return tex

    def replace(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        candidates = [name, f"{name}.tex"]
        for candidate in candidates:
            for key in (posixpath.normpath(posixpath.join(base_dir, candidate)), candidate):
                if key in files:
                    child = strip_comments(files[key])
                    return resolve_inputs(child, files, posixpath.dirname(key), depth + 1)
        return ""

    return _INPUT_RE.sub(replace, tex)


def find_main_file(files: Mapping[str, str]) -> str | None:
    candidates = [
        name for name, text in files.items() if name.endswith(".tex") and "\\documentclass" in text
    ]
    with_document = [n for n in candidates if "\\begin{document}" in files[n]]
    pool = with_document or candidates
    if not pool:
        return None
    return max(pool, key=lambda n: len(files[n]))


def _label(body: str) -> str | None:
    return command_argument(body, "label")


def _clean_math(body: str) -> str:
    body = re.sub(r"\\label\s*\{[^}]*\}", "", body)
    return body.strip()


def _display_math(tex: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for match in re.finditer(r"(?<!\\)\\\[(.+?)\\\]", tex, re.DOTALL):
        found.append((match.start(), match.group(1)))
    for match in re.finditer(r"(?<!\\)\$\$(.+?)\$\$", tex, re.DOTALL):
        found.append((match.start(), match.group(1)))
    return found


FRONT_MATTER = "Front matter"


class _SectionIndex:
    def __init__(self, sections: list[tuple[int, str]]):
        self._positions = [p for p, _ in sections]
        self._titles = [t for _, t in sections]

    def at(self, position: int) -> str | None:
        idx = bisect_right(self._positions, position) - 1
        return self._titles[idx] if idx >= 0 else None


def _sections(body: str) -> tuple[list[Section], list[tuple[int, str]]]:
    heads: list[tuple[int, int, str, int]] = []
    for match in _SECTION_RE.finditer(body):
        title, after = read_braced(body, match.end() - 1)
        heads.append(
            (match.start(), after, " ".join(title.split()), SECTION_LEVELS[match.group(1)])
        )
    sections: list[Section] = []
    lead = body[: heads[0][0] if heads else len(body)].strip()
    if lead:
        sections.append(Section(title=FRONT_MATTER, level=1, text=lead))
    for i, (_, after, title, level) in enumerate(heads):
        stop = heads[i + 1][0] if i + 1 < len(heads) else len(body)
        sections.append(Section(title=title, level=level, text=body[after:stop].strip()))
    return sections, [(start, title) for start, _, title, _ in heads]


def parse_latex(
    tex: str,
    metadata: PaperMetadata | None = None,
    files: Mapping[str, str] | None = None,
    base_dir: str = "",
) -> Paper:
    """Parse a LaTeX document (optionally with sibling files for ``\\input``) into a Paper.

    ``base_dir`` is the main file's directory within ``files``; includes resolve against it.
    """
    tex = strip_comments(tex)
    if files is not None:
        tex = resolve_inputs(tex, files, base_dir)

    metadata = metadata.model_copy() if metadata else PaperMetadata()
    if not metadata.title:
        title = command_argument(tex, "title")
        metadata.title = " ".join(title.split()) if title else ""

    abstract_envs = find_environments(tex, ("abstract",))
    if abstract_envs and not metadata.abstract:
        metadata.abstract = " ".join(abstract_envs[0].body.split())

    doc = find_environments(tex, ("document",))
    body = doc[0].body if doc else tex

    sections, heads = _sections(body)
    index = _SectionIndex(heads)

    algorithms = []
    algo_spans: list[tuple[int, int]] = []
    for env in find_environments(body, ALGORITHM_ENVS):
        algo_spans.append((env.start, env.end))
        algorithms.append(
            Algorithm(
                caption=command_argument(env.body, "caption"),
                label=_label(env.body),
                body=env.body.strip(),
                section=index.at(env.start),
            )
        )
    for env in find_environments(body, ("algorithmic",)):
        if any(s <= env.start < e for s, e in algo_spans):
            continue
        algorithms.append(Algorithm(body=env.body.strip(), section=index.at(env.start)))

    equations = [
        Equation(
            latex=_clean_math(env.body),
            label=_label(env.body),
            environment=env.name,
            section=index.at(env.start),
        )
        for env in find_environments(body, EQUATION_ENVS)
    ]
    for position, math in _display_math(body):
        equations.append(
            Equation(
                latex=_clean_math(math),
                label=_label(math),
                environment="display",
                section=index.at(position),
            )
        )

    figures = []
    for env in find_environments(body, FIGURE_ENVS):
        kind = "tikz" if "tikzpicture" in env.body else "image"
        figures.append(
            Figure(
                caption=" ".join((command_argument(env.body, "caption") or "").split()),
                label=_label(env.body),
                kind=kind,
                files=[f.strip() for f in _GRAPHICS_RE.findall(env.body)],
            )
        )

    theorems = [
        f"{env.name}: {' '.join(env.body.split())}" for env in find_environments(body, THEOREM_ENVS)
    ]

    return Paper(
        metadata=metadata,
        source_format="latex",
        sections=sections,
        equations=equations,
        algorithms=algorithms,
        figures=figures,
        theorems=theorems,
    )

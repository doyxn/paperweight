from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from paperweight.latex import find_environments, parse_latex, read_braced, strip_comments


def test_parse_sample_document(sample_tex, fixture_dir):
    files = {"sections/algo.tex": (fixture_dir / "sections/algo.tex").read_text()}
    paper = parse_latex(sample_tex, files=files)

    assert paper.metadata.title == "Fast Exponential Smoothing \\\\ for Streams"
    assert paper.metadata.abstract == "We present an online smoother with $O(1)$ updates."
    assert [s.title for s in paper.sections] == ["Method", "Experiments", "Setup"]

    labelled = [e for e in paper.equations if e.label == "eq:update"]
    assert len(labelled) == 1
    assert "\\label" not in labelled[0].latex
    assert labelled[0].section == "Method"
    envs = sorted(e.environment for e in paper.equations)
    assert envs == ["align*", "display", "equation"]
    assert all("comment" not in e.latex for e in paper.equations)

    assert len(paper.algorithms) == 1
    algo = paper.algorithms[0]
    assert algo.caption == "Online smoothing"
    assert algo.label == "alg:smooth"
    assert "\\FOR" in algo.body

    assert [(f.caption, f.kind) for f in paper.figures] == [
        ("System architecture.", "image"),
        ("Data flow.", "tikz"),
    ]
    assert paper.figures[0].files == ["figs/arch.pdf"]
    assert paper.theorems and paper.theorems[0].startswith("theorem:")


def test_missing_input_is_dropped(sample_tex):
    paper = parse_latex(sample_tex)
    assert paper.algorithms == []


def test_standalone_algorithmic_is_captured():
    tex = "\\begin{document}\\begin{algorithmic}\\STATE x\\end{algorithmic}\\end{document}"
    paper = parse_latex(tex)
    assert len(paper.algorithms) == 1
    assert paper.algorithms[0].caption is None


def test_strip_comments_keeps_escaped_percent():
    assert strip_comments("50\\% done % note") == "50\\% done "


def test_nested_same_name_environments():
    tex = "\\begin{a}x\\begin{a}y\\end{a}z\\end{a}"
    (env,) = find_environments(tex, ("a",))
    assert env.body == "x\\begin{a}y\\end{a}z"


def test_unterminated_environment_is_ignored():
    assert find_environments("\\begin{equation} x", ("equation",)) == []


_plain = st.text(alphabet=st.characters(blacklist_characters="{}\\"), max_size=20)


@st.composite
def balanced(draw, depth=0):
    parts = draw(st.lists(_plain, min_size=1, max_size=3))
    if depth < 3 and draw(st.booleans()):
        inner = draw(balanced(depth=depth + 1))
        return parts[0] + "{" + inner + "}" + "".join(parts[1:])
    return "".join(parts)


@given(balanced(), _plain)
def test_read_braced_roundtrip(content, suffix):
    text = "{" + content + "}" + suffix
    got, end = read_braced(text, 0)
    assert got == content
    assert text[end:] == suffix


@given(st.lists(_plain, min_size=1, max_size=5))
def test_find_environments_counts(bodies):
    tex = "".join(f"\\begin{{equation}}{b}\\end{{equation}}" for b in bodies)
    envs = find_environments(tex, ("equation",))
    assert [e.body for e in envs] == bodies

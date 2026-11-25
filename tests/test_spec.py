from __future__ import annotations

import json

import pytest

from paperweight.latex import parse_latex
from paperweight.llm import LLMError, extract_json
from paperweight.spec import build_paper_digest, generate_spec, render_spec_markdown


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Here:\n```json\n{"a": {"b": 2}}\n```\nDone') == {"a": {"b": 2}}
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_digest_includes_extracted_content(sample_tex):
    digest = build_paper_digest(parse_latex(sample_tex))
    assert "Fast Exponential Smoothing" in digest
    assert "eq:update" in digest
    assert "System architecture." in digest
    assert "## Paper text" in digest


def test_digest_respects_budget(sample_tex):
    assert len(build_paper_digest(parse_latex(sample_tex), budget=200)) <= 220


def test_generate_spec_retries_on_invalid(sample_tex, spec_json, fake_llm_factory):
    llm = fake_llm_factory(['{"summary": "missing project name"}', json.dumps(spec_json)])
    spec = generate_spec(parse_latex(sample_tex), llm)
    assert spec.project_name == "fast_smoothing"
    assert len(llm.calls) == 2
    assert "previous reply was invalid" in llm.calls[1][1]


def test_generate_spec_gives_up(sample_tex, fake_llm_factory):
    llm = fake_llm_factory(["nope"] * 2)
    with pytest.raises(LLMError):
        generate_spec(parse_latex(sample_tex), llm, max_attempts=2)


def test_render_markdown(spec, sample_tex):
    md = render_spec_markdown(spec, parse_latex(sample_tex))
    for needle in ("# fast_smoothing", "INV-1", "EC-1", "T-1", "SmootherState", "O(1)"):
        assert needle in md

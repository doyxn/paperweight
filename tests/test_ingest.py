from __future__ import annotations

import gzip
import io
import tarfile

import httpx
import pytest
import respx

from paperweight.ingest import (
    ARXIV_API,
    IngestError,
    load_paper,
    parse_arxiv_id,
    parse_latex_files,
    unpack_source,
)
from paperweight.models import Paper

ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Attention Is
      All You Need</title>
    <summary>  The dominant sequence transduction models...  </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
  </entry>
</feed>"""


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1706.03762", "1706.03762"),
        ("arXiv:1706.03762v7", "1706.03762v7"),
        ("https://arxiv.org/abs/1706.03762", "1706.03762"),
        ("https://arxiv.org/pdf/1706.03762v5.pdf", "1706.03762v5"),
        ("http://arxiv.org/abs/hep-th/9901001", "hep-th/9901001"),
        ("https://arxiv.org/abs/math.GT/0309136", "math.GT/0309136"),
        ("not a paper", None),
        ("https://example.com/1706.03762", None),
    ],
)
def test_parse_arxiv_id(source, expected):
    assert parse_arxiv_id(source) == expected


def _tarball(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_unpack_source_variants(sample_tex):
    kind, files = unpack_source(_tarball({"./main.tex": sample_tex, "fig.png": "x"}))
    assert kind == "latex" and list(files) == ["main.tex"]

    kind, files = unpack_source(gzip.compress(sample_tex.encode()))
    assert kind == "latex" and files == {"main.tex": sample_tex}

    assert unpack_source(b"%PDF-1.5 ...")[0] == "pdf"
    assert unpack_source(gzip.compress(b"%PDF-1.5 ..."))[0] == "pdf"

    with pytest.raises(IngestError):
        unpack_source(b"garbage")


@respx.mock
def test_load_paper_from_arxiv(sample_tex, fixture_dir):
    algo = (fixture_dir / "sections/algo.tex").read_text()
    respx.get(ARXIV_API).mock(return_value=httpx.Response(200, text=ATOM_FEED))
    respx.get("https://arxiv.org/e-print/1706.03762").mock(
        return_value=httpx.Response(
            200, content=_tarball({"main.tex": sample_tex, "sections/algo.tex": algo})
        )
    )
    paper = load_paper("https://arxiv.org/abs/1706.03762")
    assert paper.metadata.title == "Attention Is All You Need"
    assert paper.metadata.authors == ["Ashish Vaswani", "Noam Shazeer"]
    assert paper.metadata.source_url == "https://arxiv.org/abs/1706.03762"
    assert paper.source_format == "latex"
    assert len(paper.algorithms) == 1


def test_load_local_directory(fixture_dir):
    paper = load_paper(str(fixture_dir))
    assert paper.metadata.title.startswith("Fast Exponential Smoothing")
    assert len(paper.algorithms) == 1


def test_load_rejects_unknown_source():
    with pytest.raises(IngestError):
        load_paper("definitely-not-a-paper")


@respx.mock
def test_pdf_wrapper_source_falls_back_to_pdf(monkeypatch):
    wrapper = "\\documentclass{article}\\begin{document}\\includepdf{main.pdf}\\end{document}"
    respx.get(ARXIV_API).mock(return_value=httpx.Response(200, text=ATOM_FEED))
    respx.get("https://arxiv.org/e-print/1412.6980").mock(
        return_value=httpx.Response(200, content=_tarball({"arxiv.tex": wrapper}))
    )
    pdf = respx.get("https://arxiv.org/pdf/1412.6980").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.5")
    )
    seen = {}

    def fake_parse_pdf(data, metadata=None):
        seen["data"] = data
        return Paper(metadata=metadata, source_format="pdf")

    monkeypatch.setattr("paperweight.ingest.parse_pdf", fake_parse_pdf)
    paper = load_paper("1412.6980")
    assert pdf.called and seen["data"] == b"%PDF-1.5"
    assert paper.source_format == "pdf"


def test_local_tex_file_resolves_sibling_includes(fixture_dir):
    paper = load_paper(str(fixture_dir / "sample.tex"))
    assert len(paper.algorithms) == 1


def test_nested_main_resolves_includes_relative_to_it(sample_tex, fixture_dir):
    algo = (fixture_dir / "sections/algo.tex").read_text()
    kind, files = unpack_source(
        _tarball({"paper/main.tex": sample_tex, "paper/sections/algo.tex": algo})
    )
    assert len(parse_latex_files(files).algorithms) == 1


def test_local_directory_skips_symlinks_outside(tmp_path, sample_tex):
    secret = tmp_path / "secret.tex"
    secret.write_text("TOPSECRET")
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.tex").write_text(
        sample_tex.replace("\\end{document}", "\\input{leak}\\end{document}")
    )
    (src / "leak.tex").symlink_to(secret)
    paper = load_paper(str(src))
    assert "TOPSECRET" not in paper.model_dump_json()


def test_unpack_source_size_limit(monkeypatch, sample_tex):
    monkeypatch.setattr("paperweight.ingest.MAX_SOURCE_BYTES", 100)
    with pytest.raises(IngestError, match="size limit"):
        unpack_source(gzip.compress(b"\\documentclass{article}" + b" " * 1000))
    with pytest.raises(IngestError, match="size limit"):
        unpack_source(_tarball({"main.tex": sample_tex}))


@respx.mock
def test_archive_without_main_falls_back_to_pdf(monkeypatch):
    respx.get(ARXIV_API).mock(return_value=httpx.Response(200, text=ATOM_FEED))
    respx.get("https://arxiv.org/e-print/1412.6980").mock(
        return_value=httpx.Response(200, content=_tarball({"macros.tex": "\\newcommand{\\x}{y}"}))
    )
    respx.get("https://arxiv.org/pdf/1412.6980").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.5")
    )
    monkeypatch.setattr(
        "paperweight.ingest.parse_pdf",
        lambda data, metadata=None: Paper(metadata=metadata, source_format="pdf"),
    )
    assert load_paper("1412.6980").source_format == "pdf"

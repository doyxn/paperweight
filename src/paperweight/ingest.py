"""Stage 1: fetch a paper (arXiv id/URL or local file) and parse it into a Paper."""

from __future__ import annotations

import gzip
import io
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from pypdf import PdfReader

from paperweight.latex import find_main_file, parse_latex
from paperweight.models import Algorithm, Paper, PaperMetadata, Section

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_EPRINT = "https://arxiv.org/e-print/{id}"
ARXIV_PDF = "https://arxiv.org/pdf/{id}"
USER_AGENT = "paperweight/0.1 (+https://github.com/doyxn/paperweight)"
ATOM = "{http://www.w3.org/2005/Atom}"
TEXT_SUFFIXES = (".tex", ".sty", ".cls", ".bbl", ".bib")

_NEW_ID = r"\d{4}\.\d{4,5}(?:v\d+)?"
_OLD_ID = r"[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"
_ID_RE = re.compile(rf"(?:arxiv:)?({_NEW_ID}|{_OLD_ID})", re.IGNORECASE)
_URL_RE = re.compile(
    rf"arxiv\.org/(?:abs|pdf|e-print|html)/({_NEW_ID}|{_OLD_ID})(?:\.pdf)?", re.IGNORECASE
)


class IngestError(RuntimeError):
    pass


def parse_arxiv_id(source: str) -> str | None:
    source = source.strip()
    url = _URL_RE.search(source)
    if url:
        return url.group(1)
    match = _ID_RE.fullmatch(source)
    return match.group(1) if match else None


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=httpx.Timeout(60.0)
    )


def fetch_metadata(arxiv_id: str, client: httpx.Client) -> PaperMetadata:
    response = client.get(ARXIV_API, params={"id_list": arxiv_id})
    response.raise_for_status()
    root = ET.fromstring(response.text)
    entry = root.find(f"{ATOM}entry")
    if entry is None:
        raise IngestError(f"arXiv returned no entry for {arxiv_id}")

    def text(tag: str) -> str:
        node = entry.find(f"{ATOM}{tag}")
        return " ".join((node.text or "").split()) if node is not None else ""

    authors = [
        " ".join((a.findtext(f"{ATOM}name") or "").split()) for a in entry.findall(f"{ATOM}author")
    ]
    return PaperMetadata(
        arxiv_id=arxiv_id,
        title=text("title"),
        authors=[a for a in authors if a],
        abstract=text("summary"),
        source_url=f"https://arxiv.org/abs/{arxiv_id}",
    )


def _decode(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def unpack_source(data: bytes) -> tuple[str, dict[str, str] | bytes]:
    """Classify arXiv e-print bytes. Returns ("latex", files) or ("pdf", pdf_bytes)."""
    if data[:4] == b"%PDF":
        return "pdf", data
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            files: dict[str, str] = {}
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(TEXT_SUFFIXES):
                    continue
                handle = tar.extractfile(member)
                if handle is not None:
                    name = member.name[2:] if member.name.startswith("./") else member.name
                    files[name] = _decode(handle.read())
            if files:
                return "latex", files
    except tarfile.TarError:
        pass
    raw = data
    if data[:2] == b"\x1f\x8b":
        raw = gzip.decompress(data)
        if raw[:4] == b"%PDF":
            return "pdf", raw
    text = _decode(raw)
    if "\\documentclass" in text or "\\begin{document}" in text:
        return "latex", {"main.tex": text}
    raise IngestError("unrecognised arXiv source format")


def is_empty(paper: Paper) -> bool:
    """True when the LaTeX source holds no content, e.g. it only wraps a PDF via \\includepdf."""
    return not (paper.sections or paper.equations or paper.algorithms)


def parse_latex_files(files: dict[str, str], metadata: PaperMetadata | None = None) -> Paper:
    main = find_main_file(files)
    if main is None:
        raise IngestError("no LaTeX file with \\documentclass found in source")
    return parse_latex(files[main], metadata=metadata, files=files)


_ALGO_HEAD = re.compile(r"^\s*Algorithm\s+\d+[:.]?.*$", re.MULTILINE)


def parse_pdf(data: bytes, metadata: PaperMetadata | None = None) -> Paper:
    """Best-effort extraction from a PDF. LaTeX source is preferred whenever it exists."""
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages)
    metadata = metadata or PaperMetadata()
    if not metadata.title:
        info_title = (reader.metadata.title if reader.metadata else None) or ""
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        metadata = metadata.model_copy(update={"title": info_title or first_line})

    algorithms = []
    heads = list(_ALGO_HEAD.finditer(text))
    for match in heads:
        block = text[match.start() : match.start() + 2000]
        algorithms.append(Algorithm(caption=match.group(0).strip(), body=block.strip()))

    return Paper(
        metadata=metadata,
        source_format="pdf",
        sections=[Section(title="Full text", level=1, text=text)],
        algorithms=algorithms,
    )


def load_paper(source: str, client: httpx.Client | None = None) -> Paper:
    """Load a paper from an arXiv id/URL, a local PDF, .tex file, source tarball or directory."""
    path = Path(source).expanduser()
    if path.exists():
        return _load_local(path)

    arxiv_id = parse_arxiv_id(source)
    if not arxiv_id:
        raise IngestError(f"not a local file or arXiv id/URL: {source!r}")

    owns_client = client is None
    client = client or _client()
    try:
        metadata = fetch_metadata(arxiv_id, client)
        response = client.get(ARXIV_EPRINT.format(id=arxiv_id))
        if response.status_code == 200 and response.content:
            try:
                kind, payload = unpack_source(response.content)
            except IngestError:
                kind, payload = "missing", b""
            if kind == "latex":
                assert isinstance(payload, dict)
                paper = parse_latex_files(payload, metadata)
                if not is_empty(paper):
                    return paper
            if kind == "pdf":
                assert isinstance(payload, bytes)
                return parse_pdf(payload, metadata)
        pdf = client.get(ARXIV_PDF.format(id=arxiv_id))
        pdf.raise_for_status()
        return parse_pdf(pdf.content, metadata)
    finally:
        if owns_client:
            client.close()


def _load_local(path: Path) -> Paper:
    if path.is_dir():
        files = {
            str(p.relative_to(path)): _decode(p.read_bytes())
            for p in path.rglob("*")
            if p.is_file() and p.suffix in TEXT_SUFFIXES
        }
        return parse_latex_files(files)
    data = path.read_bytes()
    if path.suffix.lower() == ".pdf" or data[:4] == b"%PDF":
        return parse_pdf(data)
    if path.suffix == ".tex":
        return parse_latex(_decode(data))
    kind, payload = unpack_source(data)
    if kind == "pdf":
        assert isinstance(payload, bytes)
        return parse_pdf(payload)
    assert isinstance(payload, dict)
    return parse_latex_files(payload)

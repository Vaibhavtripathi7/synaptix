"""PDF → ordered, typed, bbox-anchored blocks using PyMuPDF's layout model.

Headers/footers are dropped (retrieval noise). Figures are rasterised by clipping the page to
the detected bbox, which also captures vector drawings that xref-based extraction misses.
Pages with no usable text are rendered whole and flagged for OCR by the enrichment step.
"""

import re
from dataclasses import dataclass, field

import pymupdf
import pymupdf4llm

from app.db.models import ElementKind

_SKIP = {"page-header", "page-footer"}
_MIN_FIGURE_PT = 40  # ignore icons / rules
_MIN_PAGE_TEXT = 40  # chars; below this a page is treated as scanned
_FIGURE_DPI = 110
_HEADING_RE = re.compile(r"^(#+)\s*(.*?)\s*$", re.S)


@dataclass(slots=True)
class Block:
    page: int  # 1-based
    role: (
        str  # layout class: title, section-header, text, list-item, caption, table, picture, scan…
    )
    kind: ElementKind
    content: str  # markdown; empty for figures until enriched
    bbox: tuple[float, float, float, float] | None
    image_png: bytes | None = None
    meta: dict = field(default_factory=dict)

    @property
    def heading_level(self) -> int | None:
        """Heading depth for section-headers, None otherwise. The layout model assigns `#` depth
        by font size, so bold and italic headings often share a depth; italic-only headings are
        pushed one level deeper, which matches the common "bold section / italic subsection"
        convention."""
        if self.role != "section-header" or not self.heading_text:
            return None
        m = _HEADING_RE.match(self.content)
        if not m:
            return 2
        raw = m.group(2)
        return len(m.group(1)) + (1 if raw.startswith("_") and not raw.startswith("**") else 0)

    @property
    def heading_text(self) -> str:
        m = _HEADING_RE.match(self.content)
        return (m.group(2) if m else self.content).strip("_* ")


@dataclass(slots=True)
class ParsedDoc:
    title: str | None
    page_count: int
    blocks: list[Block]


def parse_pdf(data: bytes) -> ParsedDoc:
    doc = pymupdf.open(stream=data, filetype="pdf")
    pages = pymupdf4llm.to_markdown(doc, page_chunks=True, write_images=False)
    blocks: list[Block] = []
    title = (doc.metadata or {}).get("title") or None

    for p in pages:
        n = p["metadata"]["page_number"]
        page = doc[n - 1]
        text = p["text"]
        page_blocks: list[Block] = []
        for box in p["page_boxes"] or []:
            cls = box["class"]
            if cls in _SKIP:
                continue
            rect = tuple(float(v) for v in box["bbox"])
            if cls == "picture":
                if rect[2] - rect[0] < _MIN_FIGURE_PT or rect[3] - rect[1] < _MIN_FIGURE_PT:
                    continue
                png = page.get_pixmap(clip=pymupdf.Rect(rect), dpi=_FIGURE_DPI).tobytes("png")
                page_blocks.append(Block(n, cls, ElementKind.figure, "", rect, png))
                continue
            start, end = box["pos"]
            content = text[start:end].strip()
            if not content:
                continue
            kind = ElementKind.table if cls == "table" else ElementKind.text
            page_blocks.append(Block(n, cls, kind, content, rect))
            if cls == "title" and not title:
                title = page_blocks[-1].heading_text

        text_chars = sum(len(b.content) for b in page_blocks if b.kind is ElementKind.text)
        if text_chars < _MIN_PAGE_TEXT:
            # Scanned or image-only page: keep a full render for OCR instead of the fragments.
            png = page.get_pixmap(dpi=_FIGURE_DPI).tobytes("png")
            page_blocks = [Block(n, "scan", ElementKind.text, "", None, png)]
        blocks.extend(page_blocks)

    doc.close()
    return ParsedDoc(title=title, page_count=len(pages), blocks=_drop_running_heads(blocks))


def explode_scans(blocks: list[Block]) -> list[Block]:
    """A scanned page is OCR'd as one Markdown blob, so the layout model never saw its headings.
    Split that Markdown back into heading and text blocks: the document regains its hierarchy,
    which sections, chunk heading paths and citation granularity all depend on. The page render
    is dropped — the source PDF is what gets displayed, so keeping it would only cost storage."""
    out: list[Block] = []
    for b in blocks:
        if b.role != "scan" or not b.content.strip():
            out.append(b)
            continue
        out.extend(_split_markdown(b.content, b.page))
    return out


_MD_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")


def _split_markdown(markdown: str, page: int) -> list[Block]:
    blocks: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            blocks.append(Block(page, "text", ElementKind.text, body, None))
        buffer.clear()

    for line in markdown.splitlines():
        if _MD_HEADING.match(line):
            flush()
            blocks.append(Block(page, "section-header", ElementKind.text, line.strip(), None))
        else:
            buffer.append(line)
    flush()
    return blocks


def _drop_running_heads(blocks: list[Block]) -> list[Block]:
    """A 'heading' that recurs on 3+ pages is a running header the layout model missed."""
    pages_by_heading: dict[str, set[int]] = {}
    for b in blocks:
        if b.role == "section-header":
            pages_by_heading.setdefault(b.heading_text.casefold(), set()).add(b.page)
    running = {h for h, pages in pages_by_heading.items() if len(pages) >= 3}
    return [
        b
        for b in blocks
        if not (b.role == "section-header" and b.heading_text.casefold() in running)
    ]

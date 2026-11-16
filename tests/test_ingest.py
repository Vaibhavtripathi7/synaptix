from app.db.models import ElementKind
from app.ingest.chunk import build_sections, chunk_sections, count_tokens
from app.ingest.parse import Block, _drop_running_heads, parse_pdf


def _h(text: str, level: int, page: int = 1) -> Block:
    return Block(page, "section-header", ElementKind.text, "#" * level + " " + text, None)


def _t(text: str, page: int = 1) -> Block:
    return Block(page, "text", ElementKind.text, text, None)


def test_parse_extracts_typed_blocks_and_flags_scans(small_pdf):
    parsed = parse_pdf(small_pdf)
    assert parsed.page_count == 3
    roles = {b.role for b in parsed.blocks}
    assert "section-header" in roles or "title" in roles
    assert any(b.kind is ElementKind.text and b.content for b in parsed.blocks)
    scans = [b for b in parsed.blocks if b.role == "scan"]
    assert [b.page for b in scans] == [3] and scans[0].image_png.startswith(b"\x89PNG")
    assert all(b.bbox is not None for b in parsed.blocks if b.role != "scan")


def test_running_heads_are_dropped():
    blocks = [_h("Running head", 2, p) for p in (1, 2, 3)] + [_h("Real", 2), _t("body")]
    kept = _drop_running_heads(blocks)
    assert [b.heading_text for b in kept if b.role == "section-header"] == ["Real"]


def test_sections_follow_heading_hierarchy_and_title_is_root():
    blocks = [
        Block(1, "title", ElementKind.text, "# Doc", None),
        _t("preamble"),
        _h("A", 2),
        _t("a1"),
        _h("A.1", 3),
        _t("a11", page=2),
        _h("B", 2),
        _t("b1", page=2),
        _h("Doc", 1),
        _t("after-repeated-title", page=3),  # title repeated as h1: not a section
    ]
    secs = build_sections(blocks, "Doc")
    assert [s.heading_path for s in secs] == [
        ["Doc"],
        ["Doc", "A"],
        ["Doc", "A", "A.1"],
        ["Doc", "B"],
        ["Doc"],
    ]
    assert (secs[2].page_start, secs[2].page_end) == (1, 2)  # heading page -> last body page


def test_chunks_respect_budget_and_isolate_tables_and_figures():
    prose = " ".join(f"sentence {i} about revenue." for i in range(400))
    table = Block(1, "table", ElementKind.table, "|a|b|\n|-|-|\n|1|2|", None, meta={"summary": "S"})
    fig = Block(1, "picture", ElementKind.figure, "A bar chart of costs.", None)
    blocks = [_h("Results", 2), _t(prose), table, fig, _t("closing remark")]
    secs = build_sections(blocks, None)
    chunks = chunk_sections(blocks, secs, max_tokens=120, overlap=20)

    assert all(c.token_count <= 120 for c in chunks)
    assert all(count_tokens(c.content) == c.token_count for c in chunks)
    kinds = [c.kind for c in chunks]
    assert kinds.count(ElementKind.table) == 1 and kinds.count(ElementKind.figure) == 1
    assert all(c.content.startswith("[Results]\n") for c in chunks)
    t = next(c for c in chunks if c.kind is ElementKind.table)
    assert "S\n\n|a|b|" in t.content and t.block_idx == [2]

    text_chunks = [c for c in chunks if c.kind is ElementKind.text]
    assert len(text_chunks) > 3
    # overlap: the tail of one prose chunk reappears at the head of the next
    tail = text_chunks[0].content.split()[-3:]
    assert " ".join(tail) in text_chunks[1].content


async def test_scanned_pages_are_transcribed_in_one_batched_call():
    """Batching is the cost lever: N scanned pages must cost one request, and each page must
    receive its own transcription even if the model returns them out of order."""
    from app.ingest.enrich import enrich
    from app.ingest.parse import ParsedDoc

    class FakeLLM:
        calls: list[list] = []

        async def generate_structured(self, content, schema, **kw):
            FakeLLM.calls.append(content)
            labels = [
                int(p.split()[1].rstrip(":"))
                for p in content
                if isinstance(p, str) and p.startswith("Page ")
            ]
            return schema(
                pages=[{"page_index": n, "markdown": f"text {n}"} for n in reversed(labels)]
            )

        @staticmethod
        def image(png: bytes):
            return {"image": png}

    scans = [Block(p, "scan", ElementKind.text, "", None, b"png") for p in range(1, 6)]
    await enrich(ParsedDoc(title=None, page_count=5, blocks=scans), FakeLLM(), batch_pages=5)

    assert len(FakeLLM.calls) == 1, "five scanned pages must cost one vision request"
    assert [b.content for b in scans] == [f"text {i}" for i in range(5)]


async def test_a_failed_batch_degrades_only_its_own_pages():
    from app.ingest.enrich import enrich
    from app.ingest.parse import ParsedDoc

    class FlakyLLM:
        def __init__(self):
            self.n = 0

        async def generate_structured(self, content, schema, **kw):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("503 overloaded")
            return schema(pages=[{"page_index": 0, "markdown": "ok"}])

        @staticmethod
        def image(png: bytes):
            return {"image": png}

    scans = [Block(p, "scan", ElementKind.text, "", None, b"png") for p in range(1, 3)]
    degraded = await enrich(
        ParsedDoc(title=None, page_count=2, blocks=scans), FlakyLLM(), batch_pages=1
    )

    assert degraded == 1
    assert "503" in scans[0].meta["enrich_error"] and scans[1].content == "ok"


def test_ocr_pages_regain_their_heading_structure():
    """A scanned page is OCR'd as one Markdown blob; splitting it back into heading and text
    blocks is what gives the document sections, chunk heading paths and per-page citations."""
    from app.ingest.parse import explode_scans

    page = Block(
        4,
        "scan",
        ElementKind.text,
        "## Power\nActive power.\n\n### Reactive\nQ lags.",
        None,
        b"png",
    )
    keep = Block(4, "text", ElementKind.text, "untouched", (1.0, 2.0, 3.0, 4.0))
    out = explode_scans([page, keep])

    assert [(b.role, b.content) for b in out[:4]] == [
        ("section-header", "## Power"),
        ("text", "Active power."),
        ("section-header", "### Reactive"),
        ("text", "Q lags."),
    ]
    assert all(b.page == 4 for b in out)
    assert all(b.image_png is None for b in out), "page renders must not be stored per block"
    assert out[-1] is keep

    secs = build_sections(explode_scans([page]), None)
    assert [s.heading_path for s in secs] == [["Power"], ["Power", "Reactive"]]

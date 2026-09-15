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
    assert (secs[2].page_start, secs[2].page_end) == (1, 2)  # heading page → last body page


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

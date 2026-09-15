from app.db.models import ElementKind
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

"""Structure-aware chunking.

Sections follow the heading hierarchy (title -> h1 -> h2 ...). Within a section, prose blocks are
packed into token-bounded windows with token overlap; each table and figure is its own chunk so
a hit maps to one concrete element. Every chunk is prefixed with its heading path, which gives
the embedding the context a bare paragraph lacks.
"""

from dataclasses import dataclass, field
from itertools import groupby

import tiktoken

from app.db.models import ElementKind
from app.ingest.parse import Block

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


@dataclass(slots=True)
class SectionDraft:
    heading_path: list[str]
    page_start: int
    page_end: int
    block_idx: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ChunkDraft:
    section_idx: int
    kind: ElementKind
    content: str
    token_count: int
    page_start: int
    page_end: int
    block_idx: list[int]


def build_sections(blocks: list[Block], title: str | None) -> list[SectionDraft]:
    root = [title] if title else []
    stack: list[tuple[int, str]] = []  # (level, heading)
    sections: list[SectionDraft] = []

    def open_section(page: int) -> SectionDraft:
        s = SectionDraft(root + [h for _, h in stack], page, page)
        sections.append(s)
        return s

    current: SectionDraft | None = None
    for i, b in enumerate(blocks):
        is_title = b.role == "title" or (title and b.heading_text.casefold() == title.casefold())
        if is_title and b.heading_text:
            root, current = [b.heading_text], None
            stack.clear()
            continue
        level = b.heading_level
        if level is not None:
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, b.heading_text))
            current = open_section(b.page)
            continue
        if current is None:
            current = open_section(b.page)
        current.block_idx.append(i)
        current.page_end = b.page
    return [s for s in sections if s.block_idx]


def chunk_sections(
    blocks: list[Block], sections: list[SectionDraft], max_tokens: int, overlap: int
) -> list[ChunkDraft]:
    chunks: list[ChunkDraft] = []
    for s_idx, s in enumerate(sections):
        prefix = " > ".join(s.heading_path)
        prefix = f"[{prefix}]\n" if prefix else ""
        # One token of slack: joining prefix and body can merge a token at the boundary.
        budget = max_tokens - count_tokens(prefix) - 1
        # Runs of prose are packed together; each table/figure becomes its own chunk.
        for is_prose, run in groupby(s.block_idx, key=lambda i: blocks[i].kind is ElementKind.text):
            idx = list(run)
            if is_prose:
                chunks.extend(_pack(blocks, idx, s_idx, prefix, budget, overlap))
                continue
            for i in idx:
                b = blocks[i]
                body = b.content if b.kind is ElementKind.figure else _table_text(b)
                if not body.strip():
                    continue
                for piece in _windows(_enc.encode(body), budget, overlap):
                    chunks.append(_draft(s_idx, b.kind, prefix, piece, [b.page], [i]))
    return chunks


def _table_text(b: Block) -> str:
    summary = b.meta.get("summary")
    return f"{summary}\n\n{b.content}" if summary else b.content


def _draft(
    s_idx: int,
    kind: ElementKind,
    prefix: str,
    toks: list[int],
    pages: list[int],
    members: list[int],
) -> ChunkDraft:
    content = prefix + _enc.decode(toks).strip()
    return ChunkDraft(
        s_idx,
        kind,
        content,
        count_tokens(content),
        min(pages),
        max(pages),
        list(dict.fromkeys(members)),
    )


_SEP = _enc.encode("\n\n")


def _pack(
    blocks: list[Block], idx: list[int], s_idx: int, prefix: str, budget: int, overlap: int
) -> list[ChunkDraft]:
    """Greedy window packing over prose blocks in token space, carrying `overlap` tokens across
    chunk boundaries. Working in tokens (not text) keeps the budget exact."""
    out: list[ChunkDraft] = []
    window: list[int] = []
    members: list[int] = []
    pages: list[int] = []
    for i in idx:
        b = blocks[i]
        for toks in _windows(_enc.encode(b.content), budget, overlap):
            if window and len(window) + len(_SEP) + len(toks) > budget:
                out.append(_draft(s_idx, ElementKind.text, prefix, window, pages, members))
                carry = window[-overlap:] if overlap + len(toks) <= budget else []
                window, members, pages = carry, members[-1:], pages[-1:]
            window += (_SEP if window else []) + toks
            members.append(i)
            pages.append(b.page)
    if window:
        out.append(_draft(s_idx, ElementKind.text, prefix, window, pages, members))
    return out


def _windows(toks: list[int], budget: int, overlap: int) -> list[list[int]]:
    """Slide a token window over an oversized block; a single window for normal ones."""
    if len(toks) <= budget:
        return [toks]
    stride = max(budget - overlap, 1)
    return [toks[i : i + budget] for i in range(0, len(toks) - overlap, stride)]

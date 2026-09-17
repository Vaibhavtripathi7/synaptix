"""Turn reranked hits into a citation-numbered context block within a token budget.

Small-to-big: chunks are precise for retrieval but a whole section is often what the answer
needs. When several hits share a section, or a section is short, the section replaces the
chunk(s); otherwise the chunk stands alone. Order is by relevance, and every passage carries
the metadata the answer will cite."""

import uuid
from dataclasses import dataclass

from app.db.models import ElementKind
from app.ingest.chunk import count_tokens
from app.retrieval.hybrid import Hit

_EXPAND_UNDER_TOKENS = 1200  # a section this small is cheaper than losing its context


@dataclass(slots=True)
class Passage:
    n: int  # citation number as it appears in the answer: [n]
    chunk_ids: list[uuid.UUID]
    document_id: uuid.UUID
    title: str
    heading_path: list[str]
    page_start: int
    page_end: int
    kind: ElementKind
    text: str
    score: float


def build_context(hits: list[Hit], budget_tokens: int) -> list[Passage]:
    passages: list[Passage] = []
    used_sections: dict[uuid.UUID, Passage] = {}
    spent = 0
    for h in hits:
        c = h.chunk
        sec = c.section
        if sec is not None and sec.id in used_sections:
            used_sections[sec.id].chunk_ids.append(c.id)  # already covered by the section
            continue
        expand = (
            sec is not None
            and c.kind is ElementKind.text
            and count_tokens(sec.content) <= _EXPAND_UNDER_TOKENS
        )
        text = sec.content if expand else c.content
        cost = count_tokens(text)
        if spent + cost > budget_tokens:
            if passages:
                break
            text, cost = c.content, c.token_count  # never return an empty context
        spent += cost
        p = Passage(
            n=len(passages) + 1,
            chunk_ids=[c.id],
            document_id=c.document_id,
            title=c.document.title or c.document.filename,
            heading_path=sec.heading_path if sec else [],
            page_start=sec.page_start if expand else c.page_start,
            page_end=sec.page_end if expand else c.page_end,
            kind=c.kind,
            text=text,
            score=h.rerank_score if h.rerank_score is not None else h.score,
        )
        passages.append(p)
        if expand:
            used_sections[sec.id] = p
    return passages


def render(passages: list[Passage]) -> str:
    """Numbered passages for a prompt, each headed with the source the answer will cite."""
    parts = []
    for p in passages:
        where = " > ".join(p.heading_path) or p.title
        pages = (
            f"p.{p.page_start}" if p.page_start == p.page_end else f"pp.{p.page_start}-{p.page_end}"
        )
        parts.append(f"[{p.n}] {p.title} — {where} ({pages}, {p.kind.value})\n{p.text}")
    return "\n\n".join(parts)

"""Hybrid search: dense ANN (pgvector, cosine) and sparse BM25-style FTS (Postgres tsvector),
each returning its own ranking, fused with reciprocal rank fusion. RRF needs no score
calibration between the two systems, which is why it is the standard choice here."""

import uuid
from dataclasses import dataclass, field

from sqlalchemy import Text, cast, func, select, text
from sqlalchemy.dialects.postgresql import TSQUERY
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Chunk

RRF_K = 60


@dataclass(slots=True)
class Hit:
    chunk: Chunk
    score: float = 0.0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None
    queries: set[int] = field(default_factory=set)  # which query variants retrieved it


async def hybrid_search(
    session: AsyncSession,
    queries: list[tuple[str, list[float]]],
    *,
    document_ids: list[uuid.UUID],
    k: int,
    dense: bool = True,
    sparse: bool = True,
) -> list[Hit]:
    """Runs dense + sparse retrieval for every (text, vector) query variant and fuses all
    rankings with RRF. Returns the top-k fused hits, chunks loaded with their section.

    `document_ids` is required and callers resolve it inside one workspace: there is no
    "search everything" mode to leak across tenants by accident."""
    if not document_ids:
        return []
    fused: dict[uuid.UUID, Hit] = {}

    def absorb(rows: list[Chunk], q_idx: int, attr: str) -> None:
        for rank, chunk in enumerate(rows, start=1):
            hit = fused.setdefault(chunk.id, Hit(chunk))
            hit.score += 1.0 / (RRF_K + rank)
            hit.queries.add(q_idx)
            if getattr(hit, attr) is None or rank < getattr(hit, attr):
                setattr(hit, attr, rank)

    base = (
        select(Chunk)
        .options(selectinload(Chunk.section), selectinload(Chunk.document))
        .where(Chunk.document_id.in_(document_ids))
    )
    # HNSW returns the ef_search nearest vectors *globally* and only then applies the WHERE.
    # When one workspace is a sliver of the table its chunks may not be in that set at all, and
    # dense search silently returns nothing. An iterative scan keeps walking the graph until
    # enough rows pass the filter; strict order because the ranks feed RRF.
    await session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))

    for q_idx, (query_text, vector) in enumerate(queries):
        if dense:
            ann = base.order_by(Chunk.embedding.cosine_distance(vector)).limit(k)
            absorb(list(await session.scalars(ann)), q_idx, "dense_rank")
        if not sparse:
            continue

        # OR, not AND. websearch_to_tsquery ANDs every term, so a natural-language question only
        # matched chunks containing all of its words — measured: zero chunks for a typical
        # question, sparse hit rate 0.19. Keyword retrieval wants BM25's behaviour: any term may
        # match and ranking sorts it out, so the parsed terms are rejoined with |.
        tsq = cast(
            func.replace(cast(func.plainto_tsquery("english", query_text), Text), "&", "|"), TSQUERY
        )
        keyword = (
            base.where(Chunk.tsv.op("@@")(tsq))
            .order_by(func.ts_rank_cd(Chunk.tsv, tsq).desc())
            .limit(k)
        )
        absorb(list(await session.scalars(keyword)), q_idx, "sparse_rank")

    return sorted(fused.values(), key=lambda h: h.score, reverse=True)[:k]

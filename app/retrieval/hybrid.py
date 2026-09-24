"""Hybrid search: dense ANN (pgvector, cosine) and sparse BM25-style FTS (Postgres tsvector),
each returning its own ranking, fused with reciprocal rank fusion. RRF needs no score
calibration between the two systems, which is why it is the standard choice here."""

import uuid
from dataclasses import dataclass, field

from sqlalchemy import func, select
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
    for q_idx, (query_text, vector) in enumerate(queries):
        ann = base.order_by(Chunk.embedding.cosine_distance(vector)).limit(k)
        absorb(list(await session.scalars(ann)), q_idx, "dense_rank")

        tsq = func.websearch_to_tsquery("english", query_text)
        sparse = (
            base.where(Chunk.tsv.op("@@")(tsq))
            .order_by(func.ts_rank_cd(Chunk.tsv, tsq).desc())
            .limit(k)
        )
        absorb(list(await session.scalars(sparse)), q_idx, "sparse_rank")

    return sorted(fused.values(), key=lambda h: h.score, reverse=True)[:k]

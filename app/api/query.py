import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    AnswererDep,
    SessionDep,
)
from app.db.models import DocStatus, Document, ElementKind
from app.retrieval.context import Passage

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[uuid.UUID] | None = None  # None = every ready document


class PassageOut(BaseModel):
    n: int
    chunk_ids: list[uuid.UUID]
    document_id: uuid.UUID
    title: str
    heading_path: list[str]
    page_start: int
    page_end: int
    kind: ElementKind
    text: str
    score: float

    @classmethod
    def from_passage(cls, p: Passage) -> "PassageOut":
        return cls(**{f: getattr(p, f) for f in cls.model_fields if hasattr(p, f)})


class QueryResponse(BaseModel):
    answer: str
    passages: list[PassageOut]
    queries: list[str]
    candidates: int


async def _searchable(session: AsyncSession, requested: list[uuid.UUID] | None) -> list[uuid.UUID]:
    """The documents a query may touch: the requested ones if all are ready, otherwise every
    ready document. An unknown id is a 404, never silently dropped."""
    stmt = select(Document.id).where(Document.status == DocStatus.ready)
    if requested:
        stmt = stmt.where(Document.id.in_(requested))
    ids = list(await session.scalars(stmt))
    if requested and len(ids) != len(set(requested)):
        raise HTTPException(404, "one or more documents were not found or are not ready")
    return ids


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, answerer: AnswererDep, session: SessionDep) -> QueryResponse:
    """Answer a question with citations: `[n]` markers in the answer refer to `passages`."""
    document_ids = await _searchable(session, body.document_ids)
    if not document_ids:
        raise HTTPException(404, "no ready documents to search")
    retrieval = await answerer.retrieve(session, body.question, document_ids)
    if not retrieval.passages:
        raise HTTPException(404, "no indexed content matched; is the document ready?")
    return QueryResponse(
        answer=await answerer.answer(body.question, retrieval.passages),
        passages=[PassageOut.from_passage(p) for p in retrieval.passages],
        queries=retrieval.queries,
        candidates=retrieval.candidates,
    )

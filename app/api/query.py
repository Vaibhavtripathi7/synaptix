import json
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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
    stream: bool = False


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
async def query(
    body: QueryRequest,
    answerer: AnswererDep,
    session: SessionDep,
):
    """Answer a question with citations. With `stream: true` the response is Server-Sent
    Events: one `passages` event, then `token` events, then `done`."""
    document_ids = await _searchable(session, body.document_ids)
    if not document_ids:
        raise HTTPException(404, "no ready documents to search")
    retrieval = await answerer.retrieve(session, body.question, document_ids)
    if not retrieval.passages:
        raise HTTPException(404, "no indexed content matched; is the document ready?")
    passages = [PassageOut.from_passage(p) for p in retrieval.passages]
    if not body.stream:
        answer = await answerer.answer(body.question, retrieval.passages)
        return QueryResponse(
            answer=answer,
            passages=passages,
            queries=retrieval.queries,
            candidates=retrieval.candidates,
        )

    async def events() -> AsyncIterator[str]:
        yield _sse("passages", [p.model_dump(mode="json") for p in passages])
        async for token in answerer.stream(body.question, retrieval.passages):
            yield _sse("token", token)
        yield _sse("done", {"queries": retrieval.queries, "candidates": retrieval.candidates})

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

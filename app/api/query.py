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
    WorkspaceDep,
)
from app.api.documents import owned_document
from app.db.models import Chunk, DocStatus, Document, Element, ElementKind
from app.db.session import end_read
from app.rag.answer import Style
from app.rag.graph import build_graph
from app.rag.summarize import summarize_document
from app.retrieval.context import Passage

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[uuid.UUID] | None = None  # None = every ready document in the workspace
    style: Style = "standard"  # how much explanation the answer should carry
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
    images: list[str] = []  # figure URLs, so a client can show what the answer cites

    @classmethod
    def from_passage(cls, p: Passage) -> "PassageOut":
        return cls(**{f: getattr(p, f) for f in cls.model_fields if hasattr(p, f)})


class QueryResponse(BaseModel):
    answer: str
    passages: list[PassageOut]
    queries: list[str]
    candidates: int


async def _searchable(
    session: AsyncSession, workspace: str, requested: list[uuid.UUID] | None
) -> list[uuid.UUID]:
    """The documents a query may touch: the requested ones if all belong to the workspace,
    otherwise every ready document in it. A foreign id is a 404, never silently dropped."""
    stmt = select(Document.id).where(
        Document.workspace_id == workspace, Document.status == DocStatus.ready
    )
    if requested:
        stmt = stmt.where(Document.id.in_(requested))
    ids = list(await session.scalars(stmt))
    if requested and len(ids) != len(set(requested)):
        raise HTTPException(404, "one or more documents were not found or are not ready")
    return ids


async def _attach_images(session: AsyncSession, passages: list[PassageOut]) -> None:
    """Fills `images` on figure passages: two queries for the whole result set, not per passage."""
    wanted = {c for p in passages if p.kind is ElementKind.figure for c in p.chunk_ids}
    if not wanted:
        return
    element_ids = await session.execute(
        select(Chunk.id, Chunk.element_ids).where(Chunk.id.in_(wanted))
    )
    by_chunk = dict(element_ids.all())
    rows = await session.execute(
        select(Element.id, Element.document_id, Element.meta["image_key"].astext)
        .where(Element.id.in_({e for ids in by_chunk.values() for e in ids}))
        .where(Element.meta.has_key("image_key"))
    )
    keys = {eid: (doc, key) for eid, doc, key in rows.all()}
    for p in passages:
        p.images = [
            f"/api/v1/documents/{keys[e][0]}/figures/{keys[e][1].rsplit('/', 1)[-1]}"
            for c in p.chunk_ids
            for e in by_chunk.get(c, [])
            if e in keys
        ]


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    workspace: WorkspaceDep,
    answerer: AnswererDep,
    session: SessionDep,
):
    """Answer a question with citations. With `stream: true` the response is Server-Sent
    Events: one `passages` event, then `token` events, then `done`."""
    document_ids = await _searchable(session, workspace, body.document_ids)
    if not document_ids:
        raise HTTPException(404, "no ready documents to search in this workspace")
    retrieval = await answerer.retrieve(session, body.question, document_ids)
    if not retrieval.passages:
        raise HTTPException(404, "no indexed content matched; is the document ready?")
    passages = [PassageOut.from_passage(p) for p in retrieval.passages]
    await _attach_images(session, passages)
    await end_read(session)  # generation can take tens of seconds; hold no locks through it
    if not body.stream:
        answer = await answerer.answer(body.question, retrieval.passages, body.style)
        return QueryResponse(
            answer=answer,
            passages=passages,
            queries=retrieval.queries,
            candidates=retrieval.candidates,
        )

    async def events() -> AsyncIterator[str]:
        yield _sse("passages", [p.model_dump(mode="json") for p in passages])
        async for token in answerer.stream(body.question, retrieval.passages, body.style):
            yield _sse("token", token)
        yield _sse("done", {"queries": retrieval.queries, "candidates": retrieval.candidates})

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/documents/{document_id}/summary")
async def summary(
    document_id: uuid.UUID,
    workspace: WorkspaceDep,
    answerer: AnswererDep,
    session: SessionDep,
    refresh: bool = False,
) -> dict:
    """Map-reduce summary of a ready document; cached on the document until `refresh=true`."""
    doc = await _ready(session, workspace, document_id)
    if doc.summary is None or refresh:
        await summarize_document(session, doc, answerer.llm)
    return {"document_id": doc.id, "summary": doc.summary}


@router.post("/documents/{document_id}/graph")
async def graph(
    document_id: uuid.UUID,
    workspace: WorkspaceDep,
    answerer: AnswererDep,
    session: SessionDep,
    refresh: bool = False,
) -> dict:
    """Entity–relation graph of a ready document; cached until `refresh=true`."""
    doc = await _ready(session, workspace, document_id)
    graph = doc.graph
    if graph is None or refresh:
        graph = await build_graph(session, doc, answerer.llm)
    return graph


async def _ready(session: AsyncSession, workspace: str, document_id: uuid.UUID) -> Document:
    doc = await owned_document(session, workspace, document_id)
    if doc.status is not DocStatus.ready:
        raise HTTPException(409, f"document is {doc.status.value}, not ready")
    return doc

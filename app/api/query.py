import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

from app.api.deps import (
    AdmissionDep,
    AnswerCacheDep,
    AnswererDep,
    ModelLimit,
    SessionDep,
    WorkspaceDep,
)
from app.api.documents import owned_document
from app.core import usage
from app.db.models import Chunk, DocStatus, Document, Element, ElementKind
from app.db.session import end_read
from app.rag.answer import Style
from app.rag.cache import cache_key
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
    usage: dict  # tokens this answer consumed, across expansion, reranking and generation
    cached: bool = False  # served from the answer cache: no model was called


async def _searchable(
    session: AsyncSession, workspace: str, requested: list[uuid.UUID] | None
) -> dict[uuid.UUID, datetime]:
    """The documents a query may touch, with their versions: the requested ones if all belong to
    the workspace, otherwise every ready document in it. A foreign id is a 404, never silently
    dropped."""
    stmt = select(Document.id, Document.updated_at).where(
        Document.workspace_id == workspace, Document.status == DocStatus.ready
    )
    if requested:
        stmt = stmt.where(Document.id.in_(requested))
    versions = dict((await session.execute(stmt)).all())
    if requested and len(versions) != len(set(requested)):
        raise HTTPException(404, "one or more documents were not found or are not ready")
    return versions


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


@router.post("/query", response_model=QueryResponse, dependencies=[ModelLimit])
async def query(
    body: QueryRequest,
    workspace: WorkspaceDep,
    answerer: AnswererDep,
    admission: AdmissionDep,
    cache: AnswerCacheDep,
    session: SessionDep,
):
    """Answer a question with citations. With `stream: true` the response is Server-Sent
    Events: one `passages` event, then `token` events, then `done`.

    The cache is consulted before admission, so a cached or coalesced answer never occupies one
    of the replica's scarce model slots."""
    versions = await _searchable(session, workspace, body.document_ids)
    if not versions:
        raise HTTPException(404, "no ready documents to search in this workspace")
    await end_read(session)  # waits and model calls below must not hold locks
    key = cache_key(workspace, versions, body.question, body.style, answerer.settings)

    cached = await cache.get(key)
    if cached is None and (leader := cache.follow(key)) is not None:
        try:  # another request here is computing this exact answer: wait for it
            cached = await asyncio.wait_for(asyncio.shield(leader), timeout=120)
        except TimeoutError:
            cached = None  # the leader is stuck; compute independently
    if cached is not None:
        return _serve_cached(cached, body.stream)

    try:
        ticket = admission.enter()
    except BaseException:
        cache.abandon(key)
        raise
    try:
        retrieval = await answerer.retrieve(session, body.question, list(versions))
        if not retrieval.passages:
            raise HTTPException(404, "no indexed content matched; is the document ready?")
        passages = [PassageOut.from_passage(p) for p in retrieval.passages]
        await _attach_images(session, passages)
        await end_read(session)  # generation can take tens of seconds; hold no locks through it
        payload = {
            "passages": [p.model_dump(mode="json") for p in passages],
            "queries": retrieval.queries,
            "candidates": retrieval.candidates,
        }
        if not body.stream:
            payload["answer"] = await answerer.answer(body.question, retrieval.passages, body.style)
            await cache.put(key, payload)
            return QueryResponse(**payload, usage=usage.current().as_dict())
    except BaseException:
        cache.abandon(key)
        raise
    finally:
        if not body.stream:
            ticket.release()

    async def events() -> AsyncIterator[str]:
        tokens: list[str] = []
        try:
            yield _sse("passages", payload["passages"])
            async for token in answerer.stream(body.question, retrieval.passages, body.style):
                tokens.append(token)
                yield _sse("token", token)
            await cache.put(key, payload | {"answer": "".join(tokens)})
            done = {"queries": retrieval.queries, "candidates": retrieval.candidates}
            yield _sse("done", done | {"usage": usage.current().as_dict()})
        except BaseException:
            cache.abandon(key)  # includes a client leaving mid-stream: never cache half an answer
            raise
        finally:
            ticket.release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
        background=BackgroundTask(ticket.release),  # if the client left before streaming began
    )


def _serve_cached(cached: dict, stream: bool):
    zero = usage.current().as_dict()
    if not stream:
        return QueryResponse(**cached, usage=zero, cached=True)

    async def events() -> AsyncIterator[str]:
        yield _sse("passages", cached["passages"])
        yield _sse("token", cached["answer"])
        done = {"queries": cached["queries"], "candidates": cached["candidates"]}
        yield _sse("done", done | {"usage": zero, "cached": True})

    return StreamingResponse(
        events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/documents/{document_id}/summary", dependencies=[ModelLimit])
async def summary(
    document_id: uuid.UUID,
    workspace: WorkspaceDep,
    answerer: AnswererDep,
    admission: AdmissionDep,
    session: SessionDep,
    refresh: bool = False,
) -> dict:
    """Map-reduce summary of a ready document; cached on the document until `refresh=true`."""
    doc = await _ready(session, workspace, document_id)
    if doc.summary is None or refresh:
        ticket = admission.enter()
        try:
            await summarize_document(session, doc, answerer.llm)
        finally:
            ticket.release()
    return {"document_id": doc.id, "summary": doc.summary}


@router.post("/documents/{document_id}/graph", dependencies=[ModelLimit])
async def graph(
    document_id: uuid.UUID,
    workspace: WorkspaceDep,
    answerer: AnswererDep,
    admission: AdmissionDep,
    session: SessionDep,
    refresh: bool = False,
) -> dict:
    """Entity–relation graph of a ready document; cached until `refresh=true`."""
    doc = await _ready(session, workspace, document_id)
    # `graph` is a deferred column; async SQLAlchemy forbids the implicit load, so ask for it.
    graph = await session.scalar(select(Document.graph).where(Document.id == doc.id))
    if graph is None or refresh:
        ticket = admission.enter()
        try:
            graph = await build_graph(session, doc, answerer.llm)
        finally:
            ticket.release()
    return graph


async def _ready(session: AsyncSession, workspace: str, document_id: uuid.UUID) -> Document:
    doc = await owned_document(session, workspace, document_id)
    if doc.status is not DocStatus.ready:
        raise HTTPException(409, f"document is {doc.status.value}, not ready")
    return doc

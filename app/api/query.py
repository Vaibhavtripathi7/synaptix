import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator

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
    # None = every ready document in the workspace. Capped: an unbounded list becomes an
    # unbounded IN clause, and a workspace holds at most `max_documents_per_workspace` anyway.
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=50)
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

    Order matters for capacity. A hit is answered from Redis before the database is touched, and
    the cache is consulted before admission, so a cached or coalesced answer never occupies one
    of the replica's scarce model slots."""

    def key_for(version: str) -> str:
        return cache_key(
            workspace, version, body.document_ids, body.question, body.style, answerer.settings
        )

    version = await cache.version(workspace)
    if version and (cached := await cache.get(key_for(version))) is not None:
        return _serve_cached(cached, body.stream)  # no database read at all

    document_ids = await _searchable(session, workspace, body.document_ids)
    if not document_ids:
        raise HTTPException(404, "no ready documents to search in this workspace")
    await end_read(session)  # waits and model calls below must not hold locks
    version = version or await cache.mint_version(workspace)
    key = key_for(version)

    cached = None
    if (leader := await cache.follow(key)) is not None:
        try:  # another request here is computing, or just computed, this answer: wait for it
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
    streaming = False  # once true, the stream below owns the slot and the leadership
    try:
        retrieval = await answerer.retrieve(session, body.question, document_ids)
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
            answer = await answerer.answer(body.question, retrieval.passages, body.style)
            payload["answer"] = _only_valid_citations(answer, passages)
            await cache.put(key, payload)
            return QueryResponse(**payload, usage=usage.current().as_dict())
        streaming = True
    except BaseException:
        cache.abandon(key)
        raise
    finally:
        if not streaming:
            ticket.release()

    started = False

    async def events() -> AsyncIterator[str]:
        nonlocal started
        started = True
        tokens: list[str] = []
        try:
            yield _sse("passages", payload["passages"])
            async for token in answerer.stream(body.question, retrieval.passages, body.style):
                tokens.append(token)
                yield _sse("token", token)
            answer = _only_valid_citations("".join(tokens), passages)
            await cache.put(key, payload | {"answer": answer})
            done = {"queries": retrieval.queries, "candidates": retrieval.candidates}
            # Tokens are already on the wire, so a streamed answer cannot be rewritten; the final
            # event names the citations that are real, and the cached copy is the cleaned one.
            done |= {"citations": _cited(answer), "usage": usage.current().as_dict()}
            yield _sse("done", done)
        except BaseException:
            cache.abandon(key)  # includes a client leaving mid-stream: never cache half an answer
            raise
        finally:
            ticket.release()

    def end() -> None:
        """Runs after the response. If the client left before streaming began, the generator
        never ran, so its slot and its leadership are given up here."""
        ticket.release()  # idempotent
        if not started:
            cache.abandon(key)  # only then: a finished stream's key may have a new leader

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
        background=BackgroundTask(end),
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


_CITATION = re.compile(r"\[(\d{1,3})\]")


def _only_valid_citations(answer: str, passages: list[PassageOut]) -> str:
    """Removes `[n]` markers that name no passage the model was shown: a fabricated citation
    points a reader at evidence that does not exist."""
    real = {p.n for p in passages}
    return _CITATION.sub(lambda m: m.group() if int(m.group(1)) in real else "", answer)


def _cited(answer: str) -> list[int]:
    return sorted({int(n) for n in _CITATION.findall(answer)})


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

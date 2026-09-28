"""Answer cache guarantees, through the real query route with a counting stand-in for the model.

Each test pins a promise with a cost attached if it broke: repeats must not reach the model,
a changed document must not be answered from before the change, a burst of identical questions
must cost one model call, and a failed or abandoned answer must never be cached."""

import asyncio
import hashlib
import secrets

import httpx
import pytest
from redis.asyncio import Redis
from sqlalchemy import text

from app.api.deps import _answerer
from app.core.limits import Admission, RateLimiter
from app.core.settings import get_settings
from app.db.models import ElementKind
from app.db.session import SessionLocal
from app.main import app
from app.rag.answer import Retrieval
from app.rag.cache import AnswerCache
from app.retrieval.context import Passage
from app.storage import LocalStorage


class CountingModel:
    """Stands in for the answerer: records calls; can be slowed down or made to fail."""

    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        self.retrievals = self.answers = 0
        self.delay, self.fail = delay, fail
        self.settings = get_settings()

    async def retrieve(self, session, question, document_ids):
        self.retrievals += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("model outage")
        passage = Passage(
            n=1,
            chunk_ids=[],
            document_id=document_ids[0],
            title="Test Doc",
            heading_path=[],
            page_start=1,
            page_end=1,
            kind=ElementKind.text,
            text="Q = wL/R",
            score=1.0,
        )
        return Retrieval(passages=[passage], queries=[question], candidates=1)

    async def answer(self, question, passages, style="standard"):
        self.answers += 1
        return f"answer to: {question}"

    async def stream(self, question, passages, style="standard"):
        self.answers += 1
        for token in ("half ", "an ", "answer"):
            yield token


@pytest.fixture
async def cache_redis():
    r = Redis.from_url(get_settings().redis_url.rsplit("/", 1)[0] + "/15")  # scratch db
    try:
        await r.ping()
    except Exception:
        pytest.skip("redis not reachable")
    yield r
    await r.flushdb()
    await r.aclose()


@pytest.fixture
async def api(seeded, cache_redis, tmp_path):
    """The query route with a ready document in a fresh workspace, and a model stand-in."""
    app.state.storage = LocalStorage(tmp_path)
    app.state.limiter = RateLimiter(cache_redis)
    app.state.admission = Admission(limit=64)
    app.state.answer_cache = AnswerCache(cache_redis, ttl_s=3600)
    headers = {"x-workspace": secrets.token_urlsafe(32)}
    wid = hashlib.sha256(headers["x-workspace"].encode()).hexdigest()
    async with SessionLocal() as s:
        await s.execute(
            text("update documents set workspace_id = :w where id = :d"), {"w": wid, "d": seeded}
        )
        await s.commit()
    model = CountingModel()
    app.dependency_overrides[_answerer] = lambda: model

    class Api:
        def __init__(self, client):
            self.client, self.model, self.doc = client, model, seeded
            self.headers, self.workspace = headers, wid

        async def ask(self, question, stream=False):
            return await self.client.post(
                "/api/v1/query", json={"question": question, "stream": stream}, headers=headers
            )

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            yield Api(client)
    finally:
        app.dependency_overrides.pop(_answerer, None)


async def test_a_repeated_question_is_answered_without_the_model(api):
    first = (await api.ask("What is the quality factor?")).json()
    again = (await api.ask("  what is the QUALITY factor  ")).json()  # same question, normalised
    assert first["cached"] is False and again["cached"] is True
    assert again["answer"] == first["answer"] and again["usage"]["calls"] == 0
    assert api.model.answers == 1


async def test_a_cache_hit_reads_nothing_from_the_database(api, monkeypatch):
    """The point of the library version: a hit is answered from Redis alone."""
    from app.api import query as query_module

    await api.ask("What is the quality factor?")

    async def no_database(*args, **kwargs):
        raise AssertionError("a cache hit queried the database")

    monkeypatch.setattr(query_module, "_searchable", no_database)
    assert (await api.ask("What is the quality factor?")).json()["cached"] is True


async def test_an_answer_about_a_deleted_document_is_never_served(api):
    await api.ask("What is resonance?")
    r = await api.client.delete(f"/api/v1/documents/{api.doc}", headers=api.headers)
    assert r.status_code == 204
    assert (await api.ask("What is resonance?")).status_code == 404, "served from a deleted library"


async def test_a_library_change_makes_old_answers_unreachable(api, cache_redis):
    """What the worker does when a document becomes ready: invalidate the workspace's library.
    Two questions, because the failure is subtle: if the next request re-minted the *same*
    version (a counter reset, a fixed token), asking Q1 would recompute Q1 — but Q2's
    pre-change answer would become reachable again and be served stale."""
    from app.rag.cache import invalidate_library

    await api.ask("What is resonance?")
    await api.ask("What is bandwidth?")
    await invalidate_library(cache_redis, api.workspace)
    assert (await api.ask("What is resonance?")).json()["cached"] is False
    assert (await api.ask("What is bandwidth?")).json()["cached"] is False, "stale answer revived"
    assert api.model.answers == 4


async def test_an_evicted_version_costs_a_miss_not_a_stale_answer(api, cache_redis):
    """The cache Redis evicts. Losing the version must fall back to the database path and mint
    a new one — never reuse an old version and resurrect old answers."""
    await api.ask("What is bandwidth?")
    await cache_redis.delete(f"lib:{api.workspace}")
    assert (await api.ask("What is bandwidth?")).json()["cached"] is False
    assert (await api.ask("What is bandwidth?")).json()["cached"] is True


async def test_a_burst_of_identical_questions_costs_one_model_call(api):
    api.model.delay = 0.3  # long enough for the burst to arrive while the first is in flight
    responses = await asyncio.gather(*(api.ask("What is bandwidth?") for _ in range(10)))
    assert {r.status_code for r in responses} == {200}
    assert api.model.retrievals == 1, "coalescing failed: each request called the model"
    assert len({r.json()["answer"] for r in responses}) == 1


async def test_a_failed_answer_is_not_cached_and_waiters_are_released(api):
    api.model.delay, api.model.fail = 0.2, True
    burst = asyncio.gather(*(api.ask("What fails?") for _ in range(3)), return_exceptions=True)
    results = await asyncio.wait_for(burst, timeout=5)  # the follower wait timeout is 120 s
    assert all(isinstance(r, RuntimeError) for r in results), "followers must not hang"
    api.model.fail = False
    assert (await api.ask("What fails?")).json()["cached"] is False


async def test_a_streamed_answer_is_cached_only_once_complete(api):
    r = await api.ask("Stream it?", stream=True)
    assert "event: done" in r.text
    again = await api.ask("Stream it?", stream=True)
    assert '"cached": true' in again.text and "half an answer" in again.text
    assert api.model.answers == 1


async def test_a_cache_outage_costs_latency_not_availability(api):
    app.state.answer_cache = AnswerCache(Redis.from_url("redis://127.0.0.1:1/0"), ttl_s=3600)
    r = await api.ask("Is the cache down?")
    assert r.status_code == 200 and r.json()["cached"] is False

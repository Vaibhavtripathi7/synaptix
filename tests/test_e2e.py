"""End to end, through the real google-genai SDK against a fake Gemini API (tests/fake_gemini.py).

Upload → worker ingestion → questions → study guides, with the model client's failover, rate-limit
handling and token accounting all running for real. No network, no quota: the conftest guard
would fail any test that tried to reach Google."""

import hashlib
import json
import secrets

import httpx
import pytest
from sqlalchemy import select, text

from app.core.limits import Admission, RateLimiter
from app.core.settings import get_settings
from app.db.models import Chunk, Element, ElementKind, Section
from app.db.session import SessionLocal
from app.main import app
from app.rag import answer as answer_module
from app.rag.cache import AnswerCache
from app.storage import LocalStorage
from app.workers.settings import analyze, ingest
from tests.fake_gemini import ANSWER, FIGURE, OCR_TEXT, USAGE, overloaded, quota_error

KEY = "AIzaTEST-e2e-key-000000000000000000"


class FakeQueue:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, tuple, dict]] = []

    async def enqueue_job(self, name: str, *args, **kwargs) -> None:
        self.jobs.append((name, args, kwargs))


@pytest.fixture
async def api(db, redis, tmp_path, fake_gemini):
    app.state.storage = LocalStorage(tmp_path)
    app.state.queue = FakeQueue()
    app.state.limiter = RateLimiter(redis)
    app.state.admission = Admission(limit=64)
    app.state.answer_cache = AnswerCache(redis, ttl_s=0)  # always exercise the model path
    app.state.cache_redis = redis
    headers = {"x-workspace": secrets.token_urlsafe(32), "x-gemini-key": KEY}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://t", headers=headers
    ) as client:
        yield client
    workspace = hashlib.sha256(headers["x-workspace"].encode()).hexdigest()
    async with SessionLocal() as s:
        await s.execute(text("delete from documents where workspace_id = :w"), {"w": workspace})
        await s.execute(text("delete from analysis_runs where workspace_id = :w"), {"w": workspace})
        await s.commit()


async def _run_queued(client) -> None:
    """Runs the job the API just queued, exactly as the arq worker would call it."""
    name, args, kwargs = client._transport.app.state.queue.jobs.pop()
    ctx = {
        "settings": get_settings(),
        "storage": app.state.storage,
        "cache_redis": app.state.cache_redis,
    }
    job = {"ingest": ingest, "analyze": analyze}[name]
    await job(ctx, *args, request_id=kwargs.get("request_id"))


async def _ingest(client, pdf: bytes) -> dict:
    r = await client.post("/api/v1/documents", files={"file": ("e2e.pdf", pdf, "application/pdf")})
    assert r.status_code == 202, r.text
    await _run_queued(client)
    return (await client.get(f"/api/v1/documents/{r.json()['id']}")).json()


def _sse(body: str) -> list[tuple[str, object]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


async def test_ingestion_end_to_end(api, fake_gemini, small_pdf):
    doc = await _ingest(api, small_pdf)
    assert doc["status"] == "ready" and doc["error"] is None

    ocr = [c for c in fake_gemini.calls if c.schema and "pages" in c.schema.get("properties", {})]
    assert len(ocr) == 1 and ocr[0].images == 1 and "Page 0:" in ocr[0].text
    figure_calls = [c for c in fake_gemini.calls if "Describe this figure" in c.text]
    assert len(figure_calls) == doc["meta"]["figures"]
    generate_calls = [c for c in fake_gemini.calls if c.method == "generateContent"]
    assert doc["meta"]["usage"]["calls"] == len(generate_calls)
    assert doc["meta"]["usage"]["input_tokens"] == USAGE["promptTokenCount"] * len(generate_calls)

    async with SessionLocal() as s:
        chunks = list(await s.scalars(select(Chunk).where(Chunk.document_id == doc["id"])))
        headings = [
            h
            for p in await s.scalars(
                select(Section.heading_path).where(Section.document_id == doc["id"])
            )
            for h in p
        ]
        figures = list(
            await s.scalars(
                select(Element.content).where(
                    Element.document_id == doc["id"], Element.kind == ElementKind.figure
                )
            )
        )
    assert len(chunks) == doc["meta"]["chunks"] > 0
    assert all(c.embedding is not None for c in chunks)
    assert "Scanned heading 0" in headings, "the OCR'd page's heading must become a section"
    assert any(OCR_TEXT in c.content for c in chunks), "OCR text must be retrievable"
    assert all(f == FIGURE for f in figures)


async def test_a_question_is_answered_end_to_end(api, fake_gemini, small_pdf):
    await _ingest(api, small_pdf)
    fake_gemini.calls.clear()

    r = await api.post("/api/v1/query", json={"question": "What drove revenue growth?"})
    body = r.json()
    assert r.status_code == 200, body
    assert body["answer"] == ANSWER and body["passages"] and not body["cached"]
    # query expansion, listwise reranking, generation: three model calls, each metered
    assert fake_gemini.methods().count("generateContent") == 2
    assert fake_gemini.methods().count("streamGenerateContent") == 1
    assert body["usage"]["calls"] == 3
    assert body["usage"]["input_tokens"] == 3 * USAGE["promptTokenCount"]
    assert body["usage"]["thinking_tokens"] == 3 * USAGE["thoughtsTokenCount"]


async def test_a_streamed_answer_arrives_in_pieces_with_its_usage_at_the_end(api, small_pdf):
    await _ingest(api, small_pdf)
    r = await api.post(
        "/api/v1/query", json={"question": "What drove revenue growth?", "stream": True}
    )
    events = _sse(r.text)
    assert [e for e, _ in events][0] == "passages" and events[-1][0] == "done"
    tokens = [data for e, data in events if e == "token"]
    assert len(tokens) > 1 and "".join(tokens) == ANSWER
    assert events[-1][1]["usage"]["calls"] == 3


async def test_a_daily_quota_fails_over_and_tokens_go_to_the_model_that_answered(fake_gemini):
    from app.core import usage

    settings = get_settings()
    primary, fallback = settings.llm_model, settings.llm_fallback_models[0]
    fake_gemini.script[primary] = [quota_error("Day")]
    ledger = usage.begin()
    answer = await answer_module.answerer_for("test-key").llm.generate("hello")
    assert answer == ANSWER
    assert [c.model for c in fake_gemini.calls] == [primary, fallback]
    assert set(ledger.by_model) == {fallback}, "tokens attributed to a model that never answered"


async def test_a_per_minute_limit_is_waited_out_on_the_same_model(fake_gemini, monkeypatch):
    from app.llm import client as client_module

    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(client_module.asyncio, "sleep", fake_sleep)
    primary = get_settings().llm_model
    fake_gemini.script[primary] = [quota_error("Minute", retry_s=7)]
    assert await answer_module.answerer_for("test-key").llm.generate("hello") == ANSWER
    assert slept == [8.0], "should wait the API's retryDelay plus a second"
    assert [c.model for c in fake_gemini.calls] == [primary, primary]


async def test_api_processes_fail_over_instead_of_sleeping(fake_gemini, monkeypatch):
    from app.llm import client as client_module

    async def no_sleep(seconds):
        raise AssertionError("an API process slept inside a request")

    monkeypatch.setattr(client_module.asyncio, "sleep", no_sleep)
    api_settings = get_settings().model_copy(update={"llm_wait_on_rate_limit": False})
    monkeypatch.setattr(answer_module, "get_settings", lambda: api_settings)
    answer_module.answerer_for.cache_clear()
    primary, fallback = api_settings.llm_model, api_settings.llm_fallback_models[0]
    fake_gemini.script[primary] = [quota_error("Minute", retry_s=30)]
    assert await answer_module.answerer_for("test-key").llm.generate("hello") == ANSWER
    assert [c.model for c in fake_gemini.calls] == [primary, fallback]


async def test_a_total_model_outage_degrades_one_page_not_the_document(api, fake_gemini, small_pdf):
    """Every model in the chain overloaded for the OCR call: the scanned page is marked degraded,
    everything else is still ingested and searchable."""
    settings = get_settings()
    for model in [settings.llm_model, *settings.llm_fallback_models]:
        fake_gemini.script[model] = [overloaded()]
    doc = await _ingest(api, small_pdf)
    assert doc["status"] == "ready"
    assert doc["meta"]["enrich_degraded"] >= 1
    assert doc["meta"]["chunks"] > 0


async def test_a_study_guide_end_to_end(api, fake_gemini, small_pdf):
    doc = await _ingest(api, small_pdf)
    r = await api.post(
        "/api/v1/analyses", json={"kind": "study_guide", "document_ids": [doc["id"]]}
    )
    assert r.status_code == 202, r.text
    await _run_queued(api)

    run = (await api.get(f"/api/v1/analyses/{r.json()['id']}")).json()
    assert run["status"] == "done", run["error"]
    assert set(run["result"]) == {"outline", "practice"}
    for stage in run["result"].values():
        numbers = {p["n"] for p in stage["passages"]}
        cited = {
            n
            for items in stage["output"].values()
            if isinstance(items, list)
            for x in items
            for n in x["evidence"]
        }
        assert cited <= numbers, "evidence must point at passages the stage actually used"
    assert run["usage"]["calls"] >= 2
    assert all(
        c.method != "batchEmbedContents" or c.model == get_settings().embedding_model
        for c in fake_gemini.calls
    )


async def test_a_document_becoming_ready_invalidates_cached_answers(api, redis, small_pdf):
    """The worker's side of invalidation, end to end: once a new document is ready, a question
    over the whole library must be answered afresh — the library it was answered from changed."""
    app.state.answer_cache = AnswerCache(redis, ttl_s=3600)
    await _ingest(api, small_pdf)
    question = {"question": "What drove revenue growth?"}
    await api.post("/api/v1/query", json=question)
    assert (await api.post("/api/v1/query", json=question)).json()["cached"] is True

    await _ingest(api, small_pdf + b"\n% second copy\n")  # new bytes, new hash: a second document
    assert (await api.post("/api/v1/query", json=question)).json()["cached"] is False


async def test_the_whole_pipeline_runs_on_object_storage(api, small_pdf):
    """API and worker sharing no disk: the upload lands in the bucket, the worker reads it back
    and writes figures there, the viewer's ranged reads stream from it, deletion empties it."""
    from app.storage import S3Storage
    from tests.conftest import s3_settings

    store = S3Storage(s3_settings())
    try:
        await store.ensure_bucket()
    except (httpx.HTTPError, RuntimeError):
        await store.close()
        pytest.skip("S3 test server not reachable (docker compose --profile s3 up -d s3)")
    app.state.storage = store
    try:
        doc = await _ingest(api, small_pdf)
        assert doc["status"] == "ready", doc["error"]
        assert await store.get(f"{doc['id']}/source.pdf") == small_pdf

        r = await api.get(f"/api/v1/documents/{doc['id']}/file", headers={"range": "bytes=0-7"})
        assert r.status_code == 206 and r.content == small_pdf[:8]
        async with SessionLocal() as s:
            keys = list(
                await s.scalars(
                    select(Element.meta["image_key"].astext).where(
                        Element.document_id == doc["id"], Element.meta.has_key("image_key")
                    )
                )
            )
        for key in keys:
            fig = await api.get(f"/api/v1/documents/{doc['id']}/figures/{key.rsplit('/', 1)[-1]}")
            assert fig.status_code == 200 and fig.content.startswith(b"\x89PNG")

        assert (await api.delete(f"/api/v1/documents/{doc['id']}")).status_code == 204
        with pytest.raises(FileNotFoundError):
            await store.get(f"{doc['id']}/source.pdf")
    finally:
        await store.delete_prefix("")
        await store.close()


async def test_no_transaction_is_open_while_figures_upload(api, small_pdf, tmp_path):
    """Figure uploads are network I/O against object storage; doing them inside the
    replace-extraction transaction would hold its locks for every round trip."""
    from app.storage import LocalStorage

    open_during_put: list[int] = []

    class Spy(LocalStorage):
        async def put(self, key, data, content_type):
            if "/figures/" in key:
                async with SessionLocal() as s:
                    n = await s.scalar(
                        text(
                            "select count(*) from pg_stat_activity where datname = "
                            "current_database() and state like 'idle in transaction%'"
                        )
                    )
                open_during_put.append(n)
            await super().put(key, data, content_type)

    app.state.storage = Spy(tmp_path / "spy")
    doc = await _ingest(api, small_pdf)
    assert doc["status"] == "ready"
    assert open_during_put, "the fixture PDF must produce at least one figure for this test"
    assert set(open_during_put) == {0}, f"transactions open during uploads: {open_during_put}"

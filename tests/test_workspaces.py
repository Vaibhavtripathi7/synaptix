"""Multi-tenancy guarantees, against the live database (skipped when it is not up).

Each test pins one promise a hosted deployment makes to anonymous users: their library is
invisible to others, their Gemini key never lands in storage or logs, filtered vector search
does not starve a tenant's results, and no request holds database locks while a model runs."""

import hashlib
import random
import secrets
import uuid

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import delete, select, text

from app.analysis import agents
from app.core.limits import BUCKETS, Admission, RateLimiter
from app.core.settings import get_settings
from app.db.models import Chunk, DocStatus, Document, ElementKind
from app.db.session import SessionLocal
from app.main import app
from app.rag import answer as answer_module
from app.rag.answer import Answerer
from app.retrieval.hybrid import hybrid_search
from app.retrieval.rerank import LLMReranker
from app.storage import LocalStorage
from tests.conftest import DIM, vec


class FakeQueue:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, tuple, dict]] = []

    async def enqueue_job(self, name: str, *args, **kwargs) -> None:
        self.jobs.append((name, args, kwargs))


def workspace(key: str | None = "AIzaTEST-user-key-00000000000000000") -> dict[str, str]:
    headers = {"x-workspace": secrets.token_urlsafe(32)}
    return headers | ({"x-gemini-key": key} if key else {})


@pytest.fixture
async def client(db, redis, tmp_path):
    app.state.storage = LocalStorage(tmp_path)
    app.state.queue = FakeQueue()
    app.state.limiter = RateLimiter(redis)
    app.state.admission = Admission(limit=64)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c
    async with SessionLocal() as s:  # everything these tests uploaded
        await s.execute(delete(Document).where(Document.filename == "iso.pdf"))
        await s.commit()


async def _upload(client, headers, pdf: bytes) -> httpx.Response:
    files = {"file": ("iso.pdf", pdf, "application/pdf")}
    return await client.post("/api/v1/documents", files=files, headers=headers)


async def test_a_workspace_cannot_see_or_touch_another_workspaces_documents(client, small_pdf):
    a, b = workspace(), workspace()
    doc = (await _upload(client, a, small_pdf)).json()

    assert [d["id"] for d in (await client.get("/api/v1/documents", headers=a)).json()] == [
        doc["id"]
    ]
    assert (await client.get("/api/v1/documents", headers=b)).json() == []
    for method, path in [
        ("GET", f"/api/v1/documents/{doc['id']}"),
        ("GET", f"/api/v1/documents/{doc['id']}/file"),
        ("DELETE", f"/api/v1/documents/{doc['id']}"),
        ("POST", f"/api/v1/documents/{doc['id']}/summary"),
    ]:
        r = await client.request(method, path, headers=b)
        assert r.status_code == 404, (method, path, r.status_code)  # absent, not forbidden
    r = await client.post(
        "/api/v1/query", json={"question": "anything?", "document_ids": [doc["id"]]}, headers=b
    )
    assert r.status_code == 404
    assert (await client.get(f"/api/v1/documents/{doc['id']}", headers=a)).status_code == 200


async def test_dedupe_is_per_workspace(client, small_pdf):
    a, b = workspace(), workspace()
    first = (await _upload(client, a, small_pdf)).json()
    again = (await _upload(client, a, small_pdf)).json()
    other = (await _upload(client, b, small_pdf)).json()
    assert again["id"] == first["id"], "same workspace, same file: deduplicated"
    assert other["id"] != first["id"], "another workspace must get its own copy, not A's"


SECRET = "AIzaSECRET-value-must-never-appear-0"


async def test_the_users_key_travels_masked_and_is_never_stored(client, small_pdf):
    doc = (await _upload(client, workspace(SECRET), small_pdf)).json()

    name, args, kwargs = client._transport.app.state.queue.jobs[-1]
    assert name == "ingest" and args[0] == doc["id"]
    assert isinstance(args[1], SecretStr) and args[1].get_secret_value() == SECRET
    assert SECRET not in repr(args), "arq logs job arguments"
    async with SessionLocal() as s:
        row = await s.scalar(
            text("select row_to_json(d)::text from documents d where id = :id"), {"id": doc["id"]}
        )
    assert SECRET not in row


async def test_tokens_are_required_long_and_stored_only_as_hashes(client, small_pdf):
    assert (await client.get("/api/v1/documents")).status_code == 401
    assert (
        await client.get("/api/v1/documents", headers={"x-workspace": "short"})
    ).status_code == 401

    headers = workspace()
    doc = (await _upload(client, headers, small_pdf)).json()
    async with SessionLocal() as s:
        stored = await s.scalar(select(Document.workspace_id).where(Document.id == doc["id"]))
    assert stored == hashlib.sha256(headers["x-workspace"].encode()).hexdigest()


async def test_nothing_is_queued_when_no_key_is_available(client, small_pdf, monkeypatch):
    keyless = get_settings().model_copy(
        update={"google_api_key": None, "gemini_backend": "api_key"}
    )
    monkeypatch.setattr(answer_module, "get_settings", lambda: keyless)
    answer_module.answerer_for.cache_clear()
    try:
        r = await _upload(client, workspace(key=None), small_pdf)
    finally:
        answer_module.answerer_for.cache_clear()
    assert r.status_code == 400 and "Gemini API key" in r.json()["detail"]
    assert client._transport.app.state.queue.jobs == []


async def test_filtered_vector_search_does_not_starve_a_tenant(db):
    """B's chunks cluster where the query points, A's sit on the far side. When the planner
    picks the HNSW index — as it does for large tenants — the index yields its ef_search nearest
    vectors globally and the tenant filter runs afterwards, so A loses results it plainly has.
    The control query shows the collapse is real here; hybrid_search must not suffer it."""
    rng = random.Random(7)

    def near(sign: float) -> list[float]:
        return [sign + rng.gauss(0, 0.3)] + [rng.gauss(0, 0.3) for _ in range(DIM - 1)]

    ids = {}
    async with SessionLocal() as s:
        for name, n, sign in (("B", 1500, 1.0), ("A", 1000, -1.0)):
            d = Document(
                workspace_id=name * 64,
                filename="hnsw.pdf",
                content_hash=uuid.uuid4().hex,
                storage_key="x",
                status=DocStatus.ready,
            )
            s.add(d)
            await s.flush()
            ids[name] = d.id
            s.add_all(
                Chunk(
                    document_id=d.id,
                    order_idx=i,
                    kind=ElementKind.text,
                    content=f"{name}{i}",
                    token_count=1,
                    page_start=1,
                    page_end=1,
                    embedding=near(sign),
                )
                for i in range(n)
            )
        await s.commit()
        await s.execute(text("ANALYZE chunks"))
        await s.commit()
        try:
            query, k = near(1.0), 10
            # Force the index plan large tenants get: no bitmap scan, no explicit sort.
            force = "SET LOCAL enable_sort = off; SET LOCAL enable_bitmapscan = off"

            for stmt in force.split("; ") + ["SET LOCAL hnsw.iterative_scan = off"]:
                await s.execute(text(stmt))
            control = await s.scalar(
                text(
                    "select count(*) from (select 1 from chunks where document_id = :d "
                    "order by embedding <=> cast(:q as vector) limit :k) t"
                ),
                {"d": ids["A"], "q": str(query), "k": k},
            )
            await s.commit()
            assert control < k, "scenario must reproduce the collapse, or the test proves nothing"

            for stmt in force.split("; "):
                await s.execute(text(stmt))
            hits = await hybrid_search(s, [("zzz", query)], document_ids=[ids["A"]], k=k)
            await s.commit()
            assert sum(h.dense_rank is not None for h in hits) == k
        finally:
            await s.execute(delete(Document).where(Document.filename == "hnsw.pdf"))
            await s.commit()


async def test_no_transaction_is_open_while_a_model_runs(seeded):
    """A read transaction held across a model call keeps its table locks for the call's
    duration; a migration queued on those locks then blocks every other query."""

    class SpyLLM:
        def __init__(self, session):
            self.session, self.calls = session, 0

        async def generate_structured(self, content, schema, **kw):
            self.calls += 1
            assert not self.session.in_transaction(), f"transaction open during {schema.__name__}"
            return schema(**{f: [] for f in schema.model_fields})

    class StubEmbedder:
        async def embed_documents(self, texts):
            return [vec(2) for _ in texts]

        async def embed_query(self, text):
            return vec(2)

    async with SessionLocal() as s:
        llm = SpyLLM(s)
        settings = get_settings().model_copy(update={"query_expansion": True})
        answerer = Answerer(llm, StubEmbedder(), LLMReranker(llm), settings)
        await answerer.retrieve(s, "isochrone drawbacks", [seeded])  # expansion + rerank
        await agents.run_agent(agents.PRACTICE, answerer, s, [seeded])  # facets + generation
        await agents.run_mapped_agent(agents.PRACTICE, answerer, s, [seeded])  # coverage mode
    assert llm.calls >= 4


async def test_a_workspace_over_its_rate_limit_gets_429_with_retry_after(client, monkeypatch):
    monkeypatch.setitem(BUCKETS, "read", (0.1, 3))  # 3 immediately, then one every 10 s
    a, b = workspace(), workspace()
    codes = [(await client.get("/api/v1/documents", headers=a)).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    limited = await client.get("/api/v1/documents", headers=a)
    assert limited.status_code == 429 and 1 <= int(limited.headers["retry-after"]) <= 10
    assert (await client.get("/api/v1/documents", headers=b)).status_code == 200, "per workspace"


async def test_a_full_replica_sheds_load_with_503_and_every_ticket_comes_back(client, seeded):
    """Model-backed requests over the per-replica cap are refused immediately, not queued; and
    a request that fails midway must still return its slot, or capacity leaks until restart."""
    from app.api.deps import _answerer

    class Boom:
        settings = get_settings()

        async def retrieve(self, *a, **k):
            raise RuntimeError("model outage")

    admission = client._transport.app.state.admission
    headers = workspace()
    wid = hashlib.sha256(headers["x-workspace"].encode()).hexdigest()
    async with SessionLocal() as s:
        await s.execute(
            text("update documents set workspace_id = :w where id = :d"), {"w": wid, "d": seeded}
        )
        await s.commit()
    app.dependency_overrides[_answerer] = lambda: Boom()
    try:
        admission.limit, admission.in_flight = 1, 1  # replica already at capacity
        r = await client.post("/api/v1/query", json={"question": "why?"}, headers=headers)
        assert r.status_code == 503 and r.headers["retry-after"] == "1"

        admission.in_flight = 0
        with pytest.raises(RuntimeError):  # the handler fails after being admitted
            await client.post("/api/v1/query", json={"question": "why?"}, headers=headers)
        assert admission.in_flight == 0, "a failed request leaked its admission slot"
    finally:
        app.dependency_overrides.pop(_answerer, None)

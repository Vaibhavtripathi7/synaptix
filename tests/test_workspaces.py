"""Multi-tenancy guarantees, against the live database (skipped when it is not up).

Each test pins one promise a hosted deployment makes to anonymous users: their library is
invisible to others, and their Gemini key never lands in storage or logs."""

import hashlib
import secrets

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import delete, select, text

from app.core.settings import get_settings
from app.db.models import Document
from app.db.session import SessionLocal
from app.main import app
from app.rag import answer as answer_module
from app.storage import LocalStorage


class FakeQueue:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, tuple, dict]] = []

    async def enqueue_job(self, name: str, *args, **kwargs) -> None:
        self.jobs.append((name, args, kwargs))


def workspace(key: str | None = "AIzaTEST-user-key-00000000000000000") -> dict[str, str]:
    headers = {"x-workspace": secrets.token_urlsafe(32)}
    return headers | ({"x-gemini-key": key} if key else {})


@pytest.fixture
async def client(db, tmp_path):
    app.state.storage = LocalStorage(tmp_path)
    app.state.queue = FakeQueue()
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

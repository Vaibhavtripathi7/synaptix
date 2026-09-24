"""Multi-tenancy guarantees, against the live database (skipped when it is not up).

Each test pins one promise a hosted deployment makes to anonymous users: their library is
invisible to others."""

import hashlib
import secrets

import httpx
import pytest
from sqlalchemy import delete, select

from app.db.models import Document
from app.db.session import SessionLocal
from app.main import app
from app.storage import LocalStorage


class FakeQueue:
    def __init__(self) -> None:
        self.jobs: list[tuple[str, tuple, dict]] = []

    async def enqueue_job(self, name: str, *args, **kwargs) -> None:
        self.jobs.append((name, args, kwargs))


def workspace() -> dict[str, str]:
    return {"x-workspace": secrets.token_urlsafe(32)}


@pytest.fixture
async def client(db, tmp_path):
    app.state.storage = LocalStorage(tmp_path)
    app.state.queue = FakeQueue()
    app.state.answerer = None  # no test here reaches the model
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

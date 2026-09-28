"""Validation at the trust boundary: what a hostile client sends in headers, filenames and
request bodies. Each case is an input someone would actually try."""

import secrets
import uuid

import httpx
import pytest

from app.core.validation import (
    GEMINI_KEY,
    REQUEST_ID,
    WORKSPACE_TOKEN,
    clean_filename,
    content_disposition,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\me\\thesis.pdf", "thesis.pdf"),
        ('evil"; filename="x.exe', "evil filename=x.exe"),
        ("line\r\nX-Injected: 1.pdf", "lineX-Injected: 1.pdf"),
        ("  notes   final .pdf  ", "notes final .pdf"),
        ("", "document.pdf"),
        ("....", "document.pdf"),
        ("\u202enatsiv.pdf", "natsiv.pdf"),  # right-to-left override, a classic filename spoof
    ],
)
def test_filenames_are_cleaned(raw, expected):
    assert clean_filename(raw) == expected


def test_long_filenames_are_capped_keeping_the_extension():
    name = clean_filename("a" * 500 + ".pdf")
    assert len(name) <= 200 and name.endswith(".pdf")


def test_content_disposition_cannot_be_broken_out_of():
    header = content_disposition('résumé"\r\nSet-Cookie: pwned=1.pdf')
    assert "\r" not in header and "\n" not in header
    assert header.count('"') == 2, "the quoted filename must stay one quoted string"
    assert "filename*=UTF-8''r%C3%A9sum%C3%A9" in header


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        (secrets.token_urlsafe(32), True),
        ("a" * 31, False),
        ("a" * 257, False),
        ("a" * 40 + " spaces", False),
        ("a" * 40 + "/../", False),
    ],
)
def test_workspace_tokens_are_whitelisted(value, ok):
    assert bool(WORKSPACE_TOKEN.fullmatch(value)) is ok


@pytest.mark.parametrize(
    ("value", "ok"),
    [
        ("AIzaSyA-0123456789abcdefghijklmnopqrstu", True),
        ("AQ.FAKE-test-key_0123456789abcdefXYZ", True),
        ("short", False),
        ("AIza" + "x" * 30 + "\nX-Evil: 1", False),
        ("AIza" + "x" * 300, False),
    ],
)
def test_gemini_keys_are_checked_for_shape(value, ok):
    assert bool(GEMINI_KEY.fullmatch(value)) is ok


def test_request_ids_are_restricted_to_log_safe_characters():
    assert REQUEST_ID.fullmatch("abc-123_x.y")
    assert not REQUEST_ID.fullmatch('x" level=critical msg="forged')
    assert not REQUEST_ID.fullmatch("a" * 65)


@pytest.fixture
async def api(db, redis, tmp_path):
    from app.core.limits import Admission, RateLimiter
    from app.main import app
    from app.rag.cache import AnswerCache
    from app.storage import LocalStorage

    app.state.storage = LocalStorage(tmp_path)
    app.state.limiter = RateLimiter(redis)
    app.state.admission = Admission(64)
    app.state.answer_cache = AnswerCache(redis, ttl_s=0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        yield c


async def test_the_api_rejects_malformed_input_at_the_edge(api):
    token = secrets.token_urlsafe(32)
    # A well-formed key, so each request below fails on the input under test and not on the
    # missing server key a CI runner has.
    good = {"x-workspace": token, "x-gemini-key": "AIzaTEST-validation-key-0000000000"}

    r = await api.get("/api/v1/documents", headers={"x-workspace": "a" * 40 + "/../../"})
    assert r.status_code == 401
    # The key is validated where it is consumed: any model-backed route.
    r = await api.post(
        "/api/v1/query", headers=good | {"x-gemini-key": "not a key"}, json={"question": "why?"}
    )
    assert r.status_code == 400
    ids = [str(uuid.uuid4()) for _ in range(51)]
    r = await api.post(
        "/api/v1/query", headers=good, json={"question": "why?", "document_ids": ids}
    )
    assert r.status_code == 422, "an unbounded id list must be refused before it reaches SQL"
    r = await api.post("/api/v1/query", headers=good, json={"question": "x" * 2001})
    assert r.status_code == 422
    r = await api.get("/healthz", headers={"x-request-id": 'forged" level=critical'})
    assert r.headers["x-request-id"] != 'forged" level=critical'

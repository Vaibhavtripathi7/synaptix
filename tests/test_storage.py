"""Blob storage: the SigV4 signer against AWS's published examples, and both backends through the
same behavioural tests (the S3 ones against versitygw, skipped when it is not running)."""

import asyncio
import hashlib
from datetime import UTC, datetime

import httpx
import pytest

from app.storage import LocalStorage, S3Storage, sigv4_headers
from tests.conftest import s3_settings

# From AWS's "Signature Version 4: authenticating requests (AWS S3)" examples.
AWS = {
    "access_key": "AKIAIOSFODNN7EXAMPLE",
    "secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "region": "us-east-1",
    "now": datetime(2013, 5, 24, tzinfo=UTC),
}
EMPTY = hashlib.sha256(b"").hexdigest()


def _signature(headers: dict[str, str]) -> str:
    return headers["authorization"].rsplit("Signature=", 1)[1]


def test_sigv4_get_object_with_range():
    h = sigv4_headers(
        "GET",
        "https://examplebucket.s3.amazonaws.com/test.txt",
        {"Range": "bytes=0-9"},
        EMPTY,
        **AWS,
    )
    assert "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date" in h["authorization"]
    assert _signature(h) == "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"


def test_sigv4_put_object_with_an_encoded_key():
    body = b"Welcome to Amazon S3."
    h = sigv4_headers(
        "PUT",
        "https://examplebucket.s3.amazonaws.com/test%24file.text",
        {"Date": "Fri, 24 May 2013 00:00:00 GMT", "x-amz-storage-class": "REDUCED_REDUNDANCY"},
        hashlib.sha256(body).hexdigest(),
        **AWS,
    )
    assert _signature(h) == "98ad721746da40c64f1a55b78f14c238d841ea1380cd77a1b5971af0ece108bd"


def test_sigv4_list_objects_canonicalises_the_query():
    h = sigv4_headers(
        "GET", "https://examplebucket.s3.amazonaws.com/?max-keys=2&prefix=J", {}, EMPTY, **AWS
    )
    assert _signature(h) == "34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7"


def test_sigv4_does_not_double_encode_an_already_encoded_query():
    """Found by the S3 test server, not the AWS vectors: a listing prefix arrives encoded."""
    encoded = sigv4_headers("GET", "https://b.s3.amazonaws.com/?prefix=doc3%2F", {}, EMPTY, **AWS)
    raw = sigv4_headers("GET", "https://b.s3.amazonaws.com/?prefix=doc3/", {}, EMPTY, **AWS)
    assert _signature(encoded) == _signature(raw)


async def _s3():
    s3 = S3Storage(s3_settings())
    try:
        await s3.ensure_bucket()
    except (httpx.HTTPError, RuntimeError):
        await s3.close()
        pytest.skip("S3 test server not reachable (docker compose --profile s3 up -d s3)")
    return s3


@pytest.fixture
async def s3():
    store = await _s3()
    yield store
    await store.delete_prefix("")
    await store.close()


@pytest.fixture(params=["local", "s3"])
async def storage(request, tmp_path):
    if request.param == "local":
        yield LocalStorage(tmp_path)
        return
    store = await _s3()
    yield store
    await store.delete_prefix("")
    await store.close()


async def _serve(response, range_header: str | None = None) -> tuple[int, dict, bytes]:
    """Runs a Response as ASGI and collects it - FileResponse reads Range from the scope."""
    headers = [(b"range", range_header.encode())] if range_header else []
    scope = {"type": "http", "method": "GET", "headers": headers, "path": "/"}
    messages = []
    delivered = False

    async def receive():
        # Like a real server: the (empty) body once, then block until the response finishes -
        # Starlette listens for a disconnect while streaming, and an instant reply spins it.
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)

    await response(scope, receive, send)
    start = messages[0]
    body = b"".join(m.get("body", b"") for m in messages[1:])
    return start["status"], {k.decode(): v.decode() for k, v in start["headers"]}, body


PDF = b"%PDF-1.7 " + bytes(range(256)) * 40


async def test_put_then_get_round_trips_bytes(storage):
    await storage.put("doc1/source.pdf", PDF, "application/pdf")
    assert await storage.get("doc1/source.pdf") == PDF


async def test_a_missing_object_is_file_not_found(storage):
    with pytest.raises(FileNotFoundError):
        await storage.get("nope/missing.pdf")


async def test_serving_supports_http_ranges(storage):
    """A PDF viewer fetches pages by range; a 50 MB deck must not be downloaded whole."""
    await storage.put("doc2/source.pdf", PDF, "application/pdf")
    status, _, body = await _serve(
        await storage.response("doc2/source.pdf", "application/pdf", None)
    )
    assert status == 200 and body == PDF
    rng = "bytes=0-9"
    status, headers, body = await _serve(
        await storage.response("doc2/source.pdf", "application/pdf", rng), rng
    )
    assert status == 206 and body == PDF[:10]
    assert headers["content-range"] == f"bytes 0-9/{len(PDF)}"


async def test_serving_a_missing_object_is_404(storage):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as err:
        await storage.response("nope/missing.png", "image/png", None)
    assert err.value.status_code == 404


async def test_delete_prefix_removes_one_document_and_nothing_else(storage):
    for key in (
        "doc3/source.pdf",
        "doc3/figures/p1_0.png",
        "doc3/figures/p2_4.png",
        "doc30/source.pdf",
    ):
        await storage.put(key, b"x", "application/octet-stream")
    await storage.delete_prefix("doc3/")
    for key in ("doc3/source.pdf", "doc3/figures/p1_0.png"):
        with pytest.raises(FileNotFoundError):
            await storage.get(key)
    assert await storage.get("doc30/source.pdf") == b"x", "a prefix must not match its neighbours"


async def test_s3_rejects_a_wrong_secret():
    """The test server validates signatures; without this, the passing S3 tests prove nothing."""
    from pydantic import SecretStr

    await (await _s3()).close()  # the server is up, and accepts the right secret
    bad = S3Storage(s3_settings(s3_secret_key=SecretStr("wrong-secret")))
    with pytest.raises(RuntimeError, match="403"):
        await bad.put("doc/source.pdf", b"x", "application/pdf")
    await bad.close()


async def test_s3_keys_with_awkward_characters_round_trip(s3):
    key = "doc4/figures/a key with spaces+plus=eq&amp;$dollar~tilde.png"
    await s3.put(key, b"bytes", "image/png")
    assert await s3.get(key) == b"bytes"

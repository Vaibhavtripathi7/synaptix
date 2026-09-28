"""Blob storage for source PDFs and extracted figures. Keys are `<document_id>/<name>`.

Two backends behind one interface:

* `LocalStorage` — a directory. Right while API and workers share a machine; wrong the moment
  they don't, because a worker cannot read a file another host received.
* `S3Storage` — any S3-compatible service (AWS S3, Cloudflare R2, GCS's S3 interface, MinIO).
  Talks the S3 REST API directly over the httpx client already in the dependency tree and signs
  requests with AWS Signature V4 (a few dozen lines of stdlib HMAC), rather than adding boto3,
  which is synchronous, or aioboto3, which brings all of botocore. The signer is checked against
  AWS's published test vector and against a real S3 implementation in the test suite.

Serving is part of the interface because it differs by backend: local files go out with
sendfile and HTTP Range support; objects are streamed through the API with Range passed through,
so a viewer can fetch one page of a 50 MB PDF, and authentication and the Content-Security-Policy
stay same-origin."""

import asyncio
import hashlib
import hmac
import shutil
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, unquote, urlsplit

import httpx
from fastapi import HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from app.core.settings import Settings


class Storage(Protocol):
    async def put(self, key: str, data: bytes, content_type: str) -> None: ...
    async def get(self, key: str) -> bytes: ...
    async def delete_prefix(self, prefix: str) -> None: ...
    async def response(
        self, key: str, media_type: str, range_header: str | None, filename: str | None = None
    ) -> Response: ...
    async def close(self) -> None: ...


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        p = (self._root / key).resolve()
        if not p.is_relative_to(self._root):
            raise ValueError(f"key escapes storage root: {key}")
        return p

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(p.write_bytes, data)

    async def get(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def delete_prefix(self, prefix: str) -> None:
        await asyncio.to_thread(shutil.rmtree, self._path(prefix), True)

    async def response(
        self, key: str, media_type: str, range_header: str | None, filename: str | None = None
    ) -> Response:
        path = self._path(key)
        if not path.is_file():
            raise HTTPException(404, "file not found")
        # FileResponse handles Range requests itself.
        return FileResponse(
            path, media_type=media_type, filename=filename, content_disposition_type="inline"
        )

    async def close(self) -> None:
        pass


def sigv4_headers(
    method: str,
    url: str,
    headers: dict[str, str],
    payload_sha256: str,
    *,
    access_key: str,
    secret_key: str,
    region: str,
    now: datetime,
    service: str = "s3",
) -> dict[str, str]:
    """AWS Signature Version 4: returns `headers` plus x-amz-date, x-amz-content-sha256 and
    Authorization. `url` must already be URI-encoded as it will be sent."""
    parts = urlsplit(url)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    day = amz_date[:8]
    signed = {k.lower(): v.strip() for k, v in headers.items()}
    signed |= {"host": parts.netloc, "x-amz-date": amz_date, "x-amz-content-sha256": payload_sha256}
    names = sorted(signed)
    # Decode, then encode once: the URL may already carry percent-encoded values (a listing
    # prefix `a%2F`), and encoding those again (`a%252F`) signs a different request.
    query = "&".join(
        sorted(
            f"{quote(unquote(k), safe='-_.~')}={quote(unquote(v), safe='-_.~')}"
            for k, _, v in (p.partition("=") for p in parts.query.split("&") if p)
        )
    )
    canonical = "\n".join(
        [
            method,
            parts.path or "/",
            query,
            "".join(f"{n}:{signed[n]}\n" for n in names),
            ";".join(names),
            payload_sha256,
        ]
    )
    scope = f"{day}/{region}/{service}/aws4_request"
    to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()]
    )
    key = f"AWS4{secret_key}".encode()
    for part in (day, region, service, "aws4_request"):
        key = hmac.new(key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(key, to_sign.encode(), hashlib.sha256).hexdigest()
    signed["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={';'.join(names)}, Signature={signature}"
    )
    return signed


_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


class S3Storage:
    """Path-style addressing (`<endpoint>/<bucket>/<key>`), which every S3-compatible service
    accepts; one pooled httpx client per process."""

    def __init__(self, settings: Settings) -> None:
        if not (settings.s3_bucket and settings.s3_access_key and settings.s3_secret_key):
            raise RuntimeError(
                "S3 storage needs SYNAPTIX_S3_BUCKET, _S3_ACCESS_KEY, _S3_SECRET_KEY"
            )
        self._endpoint = settings.s3_endpoint_url.rstrip("/")
        self._bucket = settings.s3_bucket
        self._region = settings.s3_region
        self._access = settings.s3_access_key
        self._secret = settings.s3_secret_key.get_secret_value()
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(60, connect=5))

    def _url(self, key: str = "", query: str = "") -> str:
        path = f"/{self._bucket}" + (f"/{quote(key, safe='/-_.~')}" if key else "")
        return f"{self._endpoint}{path}" + (f"?{query}" if query else "")

    def _signed(
        self, method: str, url: str, body: bytes = b"", headers: dict[str, str] | None = None
    ) -> dict[str, str]:
        digest = hashlib.sha256(body).hexdigest() if body else _EMPTY_SHA256
        return sigv4_headers(
            method,
            url,
            headers or {},
            digest,
            access_key=self._access,
            secret_key=self._secret,
            region=self._region,
            now=datetime.now(UTC),
        )

    async def _send(self, method: str, url: str, body: bytes = b"", headers=None) -> httpx.Response:
        r = await self._http.request(
            method, url, content=body, headers=self._signed(method, url, body, headers)
        )
        if r.status_code >= 300 and r.status_code != 404:
            raise RuntimeError(
                f"S3 {method} {urlsplit(url).path} → {r.status_code}: {r.text[:300]}"
            )
        return r

    async def ensure_bucket(self) -> None:
        """Development convenience (MinIO): create the bucket if it is missing. In production the
        bucket belongs to infrastructure and this is a cheap no-op."""
        r = await self._http.request("PUT", self._url(), headers=self._signed("PUT", self._url()))
        if r.status_code not in (200, 409):  # 409: already exists / already owned
            raise RuntimeError(
                f"S3 could not create bucket {self._bucket}: {r.status_code} {r.text[:200]}"
            )

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        await self._send("PUT", self._url(key), data, {"content-type": content_type})

    async def get(self, key: str) -> bytes:
        r = await self._send("GET", self._url(key))
        if r.status_code == 404:
            raise FileNotFoundError(key)
        return r.content

    async def delete_prefix(self, prefix: str) -> None:
        """ListObjectsV2, then a DELETE per key: a document holds one PDF and a few dozen
        figures, so this beats building DeleteObjects XML with its mandatory checksum."""
        keys: list[str] = []
        token = ""
        while True:
            query = f"list-type=2&prefix={quote(prefix, safe='')}" + (
                f"&continuation-token={quote(token, safe='')}" if token else ""
            )
            root = ET.fromstring((await self._send("GET", self._url(query=query))).content)
            keys += [k.text for k in root.iter(f"{_S3_NS}Key")]
            token = root.findtext(f"{_S3_NS}NextContinuationToken") or ""
            if not token:
                break
        await asyncio.gather(*(self._send("DELETE", self._url(k)) for k in keys))

    async def response(
        self, key: str, media_type: str, range_header: str | None, filename: str | None = None
    ) -> Response:
        url = self._url(key)
        extra = {"range": range_header} if range_header else {}
        request = self._http.build_request(
            "GET", url, headers=self._signed("GET", url, headers=extra)
        )
        upstream = await self._http.send(request, stream=True)
        if upstream.status_code == 404:
            await upstream.aclose()
            raise HTTPException(404, "file not found")
        if upstream.status_code not in (200, 206):
            await upstream.aclose()
            raise HTTPException(502, "storage error")
        headers = {
            h: upstream.headers[h]
            for h in ("content-length", "content-range")
            if h in upstream.headers
        }
        headers["accept-ranges"] = "bytes"
        if filename:
            headers["content-disposition"] = f'inline; filename="{filename}"'

        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes(64 * 1024):
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            body(), status_code=upstream.status_code, media_type=media_type, headers=headers
        )

    async def close(self) -> None:
        await self._http.aclose()


async def build_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "s3":
        storage = S3Storage(settings)
        await storage.ensure_bucket()
        return storage
    return LocalStorage(settings.storage_dir)

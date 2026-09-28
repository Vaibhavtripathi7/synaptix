"""Per-request context, as plain ASGI middleware.

Starlette's `BaseHTTPMiddleware` (what `@app.middleware("http")` builds) runs every request
through an extra task group and memory stream; on the hottest path in the app that cost is paid
per request, so this is written against the raw ASGI interface instead. It gives every request:
an id on every log line (echoed to the client), the optional operator-key gate, a token ledger,
bounded-cardinality metrics, and one structured access-log line."""

import hmac
import ipaddress
import json
import time
import uuid

import structlog

from app.core import usage
from app.core.metrics import HTTP_IN_FLIGHT, HTTP_LATENCY, HTTP_REQUESTS
from app.core.validation import REQUEST_ID

log = structlog.get_logger()


class RequestContext:
    def __init__(self, app, operator_keys: str = "", trusted_proxy_hops: int = 0) -> None:
        self.app = app
        self._hops = trusted_proxy_hops
        # Parsed once; compared in constant time so response timing reveals nothing about a key.
        self._keys = [k.strip().encode() for k in operator_keys.split(",") if k.strip()]

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        # A client-supplied id goes into every log line and a response header: accepted only if
        # it matches a strict pattern, otherwise replaced.
        supplied = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied if REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex[:12]
        scope.setdefault("state", {})["client_ip"] = self._client_ip(scope, headers)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        is_api = scope["path"].startswith("/api/")

        if is_api and self._keys and not self._authorised(headers):
            await _json(send, 401, {"detail": "invalid or missing API key"}, request_id)
            return

        ledger = usage.begin()
        status = 500
        started = time.perf_counter()
        HTTP_IN_FLIGHT.inc()

        async def send_with_context(message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message["headers"] = [
                    *message.get("headers", []),
                    (b"x-request-id", request_id.encode()),
                ]
                HTTP_LATENCY.labels(_route(scope), scope["method"]).observe(
                    time.perf_counter() - started
                )
            await send(message)

        try:
            await self.app(scope, receive, send_with_context)
        finally:
            HTTP_IN_FLIGHT.dec()
            route = _route(scope)
            HTTP_REQUESTS.labels(route, scope["method"], str(status)).inc()
            if is_api:  # total time including a streamed body, unlike the histogram
                fields = {
                    "input_tokens": ledger.input_tokens,
                    "output_tokens": ledger.output_tokens,
                }
                log.info(
                    "http",
                    method=scope["method"],
                    route=route,
                    status=status,
                    ms=round((time.perf_counter() - started) * 1000),
                    **(fields if ledger.calls else {}),
                )

    def _client_ip(self, scope, headers: dict[bytes, bytes]) -> str | None:
        """The address rate limits key on. Each trusted proxy appends the address it received
        the request from to X-Forwarded-For, so the client is `hops` entries from the right;
        anything further left was written by the client and is not believed."""
        peer = (scope.get("client") or (None,))[0]
        if self._hops <= 0:
            return peer
        forwarded = headers.get(b"x-forwarded-for", b"").decode("latin-1")
        chain = [h.strip() for h in forwarded.split(",") if h.strip()]
        if len(chain) < self._hops:
            return peer
        try:
            return str(ipaddress.ip_address(chain[-self._hops]))
        except ValueError:
            return peer

    def _authorised(self, headers: dict[bytes, bytes]) -> bool:
        token = headers.get(b"authorization", b"").removeprefix(b"Bearer ").strip()
        return any(hmac.compare_digest(token, key) for key in self._keys)


def current_request_id() -> str | None:
    """Carried into background jobs, so a request and the work it queued share one id in logs."""
    return structlog.contextvars.get_contextvars().get("request_id")


def _route(scope) -> str:
    """The matched route's template, never the raw path: bounded label values."""
    route = scope.get("route")
    return getattr(route, "path", None) or "unmatched"


async def _json(send, status: int, body: dict, request_id: str) -> None:
    payload = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
                (b"x-request-id", request_id.encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})

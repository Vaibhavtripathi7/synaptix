import hmac
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.api import analyses, documents, query
from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.db.session import engine
from app.rag.answer import build_answerer
from app.storage import LocalStorage

log = structlog.get_logger()
# Operator keys gate /api/* when set. Compared in constant time: response timing reveals nothing.
_KEYS = [k.strip().encode() for k in get_settings().api_keys.split(",") if k.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    app.state.storage = LocalStorage(settings.storage_dir)
    app.state.answerer = build_answerer(settings)
    app.state.queue = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    yield
    await app.state.queue.aclose()
    await engine.dispose()


app = FastAPI(title="Synaptix", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """A request id on every log line (echoed to the client), the optional operator-key gate,
    and one structured access-log line per API call."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    is_api = request.url.path.startswith("/api/")
    if is_api and _KEYS:
        token = request.headers.get("authorization", "").removeprefix("Bearer ").strip().encode()
        if not any(hmac.compare_digest(token, key) for key in _KEYS):
            return JSONResponse(
                {"detail": "invalid or missing API key"},
                status_code=401,
                headers={"x-request-id": request_id},
            )
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    if is_api:
        log.info(
            "http",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=round((time.perf_counter() - started) * 1000),
        )
    return response


app.include_router(documents.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(analyses.router, prefix="/api/v1")


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
async def readyz() -> dict:
    """Readiness = both backing services answer. Liveness (`/healthz`) never touches them."""
    async with engine.connect() as conn:
        await conn.execute(text("select 1"))
    await app.state.queue.ping()
    return {"status": "ready"}


# Last: the UI is a static single page; mounted after the API and probes so it never shadows them.
app.mount("/", StaticFiles(directory="web", html=True), name="ui")

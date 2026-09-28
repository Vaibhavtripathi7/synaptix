from contextlib import asynccontextmanager

import structlog
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from prometheus_client import start_http_server
from redis.asyncio import Redis
from sqlalchemy import text

from app.api import analyses, documents, query
from app.core.limits import Admission, RateLimiter
from app.core.logging import configure_logging
from app.core.middleware import RequestContext
from app.core.settings import get_settings
from app.db.session import engine
from app.rag.cache import AnswerCache
from app.storage import build_storage

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    app.state.storage = await build_storage(settings)
    app.state.queue = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.limiter = RateLimiter(app.state.queue)
    app.state.admission = Admission(settings.max_inflight_model_requests)
    app.state.cache_redis = Redis.from_url(settings.cache_redis_url)
    app.state.answer_cache = AnswerCache(app.state.cache_redis, settings.answer_cache_ttl_s)
    # Metrics on an internal port: scraped over the private network, never published. A busy
    # port (a second local process) costs the metrics endpoint, not the service.
    try:
        start_http_server(settings.metrics_port)
    except OSError as exc:
        log.warning("metrics.unavailable", port=settings.metrics_port, error=str(exc))
    yield
    await app.state.queue.aclose()
    await app.state.storage.close()
    await app.state.cache_redis.aclose()
    await engine.dispose()


app = FastAPI(title="Synaptix", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    RequestContext,
    operator_keys=get_settings().api_keys,
    trusted_proxy_hops=get_settings().trusted_proxy_hops,
)
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

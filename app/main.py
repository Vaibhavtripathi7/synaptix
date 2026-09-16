from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI
from sqlalchemy import text

from app.api import documents
from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.db.session import engine
from app.storage import LocalStorage


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    app.state.storage = LocalStorage(settings.storage_dir)
    app.state.queue = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    yield
    await app.state.queue.aclose()
    await engine.dispose()


app = FastAPI(title="Synaptix", version="0.1.0", lifespan=lifespan)
app.include_router(documents.router, prefix="/api/v1")


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

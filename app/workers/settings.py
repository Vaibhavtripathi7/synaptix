"""arq worker: `arq app.workers.settings.WorkerSettings`."""

import uuid
from collections.abc import Awaitable, Callable

import structlog
from arq.connections import RedisSettings

from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.ingest.pipeline import ingest_document
from app.llm.client import Gemini
from app.llm.embeddings import build_embedder
from app.storage import LocalStorage

log = structlog.get_logger()

JOB_TIMEOUT = 1800


async def startup(ctx: dict) -> None:
    configure_logging()
    settings = get_settings()
    llm = Gemini(settings)
    ctx.update(
        settings=settings,
        storage=LocalStorage(settings.storage_dir),
        llm=llm,
        embedder=build_embedder(settings, llm),
    )


async def _run_job(name: str, work: Callable[[], Awaitable[None]]) -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(job=name)
    await work()
    log.info("job.done")


async def ingest(ctx: dict, document_id: str) -> None:
    await _run_job(
        "ingest",
        lambda: ingest_document(
            uuid.UUID(document_id),
            storage=ctx["storage"],
            llm=ctx["llm"],
            embedder=ctx["embedder"],
            settings=ctx["settings"],
        ),
    )


class WorkerSettings:
    functions = [ingest]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 2  # parsing is CPU-bound; LLM stages are bounded by llm_concurrency
    job_timeout = JOB_TIMEOUT
    max_tries = 1  # a failed job is surfaced on its row, not retried blindly

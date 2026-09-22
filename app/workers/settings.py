"""arq worker: `arq app.workers.settings.WorkerSettings`."""

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import structlog
from arq.connections import RedisSettings
from sqlalchemy import update

from app.analysis.pipeline import run_analysis
from app.core.logging import configure_logging
from app.core.settings import get_settings
from app.db.models import DocStatus, Document
from app.db.session import SessionLocal
from app.ingest.pipeline import ingest_document
from app.rag.answer import build_answerer
from app.storage import LocalStorage

log = structlog.get_logger()

JOB_TIMEOUT = 1800

# States a job passes through. A document sitting in one of these with nothing working on it was
# abandoned by a worker that died mid-ingest.
_TRANSIENT = (DocStatus.parsing, DocStatus.enriching, DocStatus.embedding)


async def reap_stranded() -> None:
    """Fail documents abandoned mid-ingest. Only those untouched for longer than a job could
    run are considered stranded, so this never clobbers a document another worker is handling."""
    cutoff = datetime.now(UTC) - timedelta(seconds=JOB_TIMEOUT)
    async with SessionLocal() as session:
        result = await session.execute(
            update(Document)
            .where(Document.status.in_(_TRANSIENT), Document.updated_at < cutoff)
            .values(status=DocStatus.failed, error="ingestion was interrupted; upload again")
        )
        await session.commit()
    if result.rowcount:
        log.warning("ingest.reaped_stranded", documents=result.rowcount)


async def startup(ctx: dict) -> None:
    configure_logging()
    settings = get_settings()
    ctx.update(
        settings=settings,
        storage=LocalStorage(settings.storage_dir),
        answerer=build_answerer(settings),
    )
    await reap_stranded()


async def _run_job(name: str, work: Callable[[], Awaitable[None]]) -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(job=name)
    await work()
    log.info("job.done")


async def ingest(ctx: dict, document_id: str) -> None:
    answerer = ctx["answerer"]
    await _run_job(
        "ingest",
        lambda: ingest_document(
            uuid.UUID(document_id),
            storage=ctx["storage"],
            llm=answerer.llm,
            embedder=answerer.embedder,
            settings=ctx["settings"],
        ),
    )


async def analyze(ctx: dict, run_id: str) -> None:
    await _run_job("analyze", lambda: run_analysis(uuid.UUID(run_id), ctx["answerer"]))


class WorkerSettings:
    functions = [ingest, analyze]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 2  # parsing is CPU-bound; LLM stages are bounded by llm_concurrency
    job_timeout = JOB_TIMEOUT
    max_tries = 1  # a failed job is surfaced on its row, not retried blindly

"""arq worker: `arq app.workers.settings.WorkerSettings`."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import structlog
from arq import cron
from arq.connections import RedisSettings
from prometheus_client import start_http_server
from pydantic import SecretStr
from redis.asyncio import Redis
from sqlalchemy import delete, update

from app.analysis.pipeline import run_analysis
from app.core import usage
from app.core.logging import configure_logging
from app.core.metrics import JOB_LATENCY, JOBS, QUEUE_DEPTH
from app.core.settings import get_settings
from app.db.models import DocStatus, Document
from app.db.session import SessionLocal
from app.ingest.pipeline import ingest_document
from app.rag.answer import answerer_for
from app.rag.cache import invalidate_library
from app.storage import build_storage

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


async def purge_failed(ctx: dict) -> None:
    """Failed documents keep their source so a re-upload can retry. After a day nobody is
    retrying — and on an anonymous service, an upload that fails after its file is stored (a fake
    key, say) is the cheapest way to fill a disk. Rows go first, then the files: a crash between
    the two leaves orphaned files, never rows pointing at nothing."""
    cutoff = datetime.now(UTC) - timedelta(hours=ctx["settings"].failed_retention_hours)
    async with SessionLocal() as session:
        purged = (
            (
                await session.execute(
                    delete(Document)
                    .where(Document.status == DocStatus.failed, Document.updated_at < cutoff)
                    .returning(Document.id)
                )
            )
            .scalars()
            .all()
        )
        await session.commit()
    for document_id in purged:
        await ctx["storage"].delete_prefix(f"{document_id}/")
    if purged:
        log.info("purged_failed_documents", documents=len(purged))


async def startup(ctx: dict) -> None:
    configure_logging()
    settings = get_settings()
    # No model clients here: each job brings its user's key and gets that key's stack.
    ctx.update(
        settings=settings,
        storage=await build_storage(settings),
        cache_redis=Redis.from_url(settings.cache_redis_url),  # to invalidate cached answers
    )
    start_http_server(settings.worker_metrics_port)
    ctx["depth_poller"] = asyncio.create_task(_poll_queue_depth(ctx["redis"]))
    await reap_stranded()


async def shutdown(ctx: dict) -> None:
    ctx["depth_poller"].cancel()
    await ctx["cache_redis"].aclose()
    await ctx["storage"].close()


async def _poll_queue_depth(redis, every: float = 5.0) -> None:
    """Queue depth is the autoscaling signal for workers, so export it continuously, not only
    when a job happens to start."""
    while True:
        QUEUE_DEPTH.set(await redis.zcard("arq:queue"))
        await asyncio.sleep(every)


async def _run_job(name: str, request_id: str | None, work: Callable[[], Awaitable[None]]) -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(job=name, request_id=request_id)
    ledger = usage.begin()
    started = time.perf_counter()
    try:
        await work()
    except asyncio.CancelledError:
        JOBS.labels(name, "cancelled").inc()
        raise
    except Exception:
        JOBS.labels(name, "failed").inc()
        raise
    else:
        JOBS.labels(name, "ok").inc()
    finally:
        JOB_LATENCY.labels(name).observe(time.perf_counter() - started)
        log.info("job.done", tokens_in=ledger.input_tokens, tokens_out=ledger.output_tokens)


def _stack(api_key: SecretStr | None):
    return answerer_for(api_key.get_secret_value() if api_key else None)


async def ingest(
    ctx: dict, document_id: str, api_key: SecretStr | None = None, request_id: str | None = None
) -> None:
    answerer = _stack(api_key)

    async def work() -> None:
        workspace = await ingest_document(
            uuid.UUID(document_id),
            storage=ctx["storage"],
            llm=answerer.llm,
            embedder=answerer.embedder,
            settings=ctx["settings"],
        )
        if workspace:  # the document is committed as ready: its library just changed
            await invalidate_library(ctx["cache_redis"], workspace)

    await _run_job("ingest", request_id, work)


async def analyze(
    ctx: dict, run_id: str, api_key: SecretStr | None = None, request_id: str | None = None
) -> None:
    await _run_job("analyze", request_id, lambda: run_analysis(uuid.UUID(run_id), _stack(api_key)))


class WorkerSettings:
    functions = [ingest, analyze]
    cron_jobs = [cron(purge_failed, minute={7, 37})]  # twice an hour, off the top of the hour
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 2  # parsing is CPU-bound; LLM stages are bounded by llm_concurrency
    job_timeout = JOB_TIMEOUT
    max_tries = 1  # a failed job is surfaced on its row, not retried blindly
    # Job results include the job's arguments — a user's API key among them. Keep none.
    keep_result = 0

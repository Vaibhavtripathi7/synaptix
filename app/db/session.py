from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.settings import get_settings

_settings = get_settings()

# Pool tuning, each choice measured with bench/run.py:
#
# * `prepared_statement_cache_size=0` — asyncpg caches prepared statements per connection, and a
#   migration applied while workers run invalidates them: every pooled connection then raises
#   InvalidCachedStatementError until recycled. It is also mandatory behind pgbouncer in
#   transaction mode. Cost: ~12% of per-process read throughput. Kept, for safe rolling deploys.
# * no `pool_pre_ping` — it adds a round trip to *every* checkout to catch the rare dead
#   connection. Instead a dead connection fails its one request, SQLAlchemy invalidates the pool,
#   and connections are recycled on age. One failed request per database failover, in exchange
#   for a round trip saved on every request.
# * pool size — sized for the concurrency one replica actually runs. Total connections are
#   replicas × (pool + overflow); past Postgres's limit, put pgbouncer in front.
engine = create_async_engine(
    _settings.database_url,
    pool_size=_settings.db_pool_size,
    max_overflow=_settings.db_max_overflow,
    pool_recycle=1800,
    connect_args={"prepared_statement_cache_size": 0},
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def end_read(session: AsyncSession) -> None:
    """Close the current transaction before slow work that doesn't touch the database — an LLM
    call. An open transaction keeps its table locks; a migration waiting on those locks then
    queues every later query behind it, and the whole app stalls for as long as the model takes.
    Loaded objects stay usable afterwards because the session doesn't expire them on commit."""
    await session.commit()


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request, committed on success."""
    async with SessionLocal() as session:
        yield session
        await session.commit()

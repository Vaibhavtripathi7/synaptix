from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.settings import get_settings

_settings = get_settings()

# asyncpg caches prepared statements per connection. A migration applied while workers run
# invalidates them, and every pooled connection then raises InvalidCachedStatementError until
# it is recycled; the cache is also incompatible with pgbouncer in transaction mode. Off.
engine = create_async_engine(
    _settings.database_url,
    pool_pre_ping=True,
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

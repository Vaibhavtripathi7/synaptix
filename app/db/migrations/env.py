import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.core.settings import get_settings
from app.db.models import Base

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url)
target_metadata = Base.metadata


def _run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        # A DDL lock request queues every later query behind it: if a long transaction holds the
        # table, fail fast and retry the deploy rather than freeze the running app.
        #
        # This must run *inside* Alembic's transaction. Issued before it, the statement
        # auto-begins a transaction that Alembic then treats as the caller's, never commits, and
        # the connection closes with a rollback: every migration silently undone, exit code 0.
        connection.exec_driver_sql("SET LOCAL lock_timeout = '5s'")
        context.run_migrations()


async def run_async() -> None:
    engine = async_engine_from_config(config.get_section(config.config_ini_section, {}))
    async with engine.connect() as connection:
        await connection.run_sync(_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(run_async())

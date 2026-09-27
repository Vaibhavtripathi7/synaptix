"""Migrations against a throwaway database (skipped when Postgres is not reachable).

A development database already holds the schema, so it hides a migration that does nothing: one
such bug here rolled back every migration while exiting 0, and only a fresh database showed it.
Each test creates its own database and drops it, so the down/up cycle never touches real data."""

import os
import secrets
import subprocess
import sys

import asyncpg
import pytest

from app.core.settings import get_settings

EXPECTED = {"documents", "elements", "sections", "chunks", "analysis_runs"}


def _alembic(url: str, *args: str) -> str:
    env = os.environ | {"SYNAPTIX_DATABASE_URL": url}
    done = subprocess.run(
        [sys.executable, "-m", "alembic", *args], env=env, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stderr
    return done.stdout + done.stderr


@pytest.fixture
async def scratch_db(db):
    admin_url = get_settings().database_url.replace("+asyncpg", "").rsplit("/", 1)[0] + "/postgres"
    name = f"synaptix_migtest_{secrets.token_hex(4)}"
    admin = await asyncpg.connect(admin_url)
    await admin.execute(f'CREATE DATABASE "{name}"')
    try:
        yield (
            get_settings().database_url.rsplit("/", 1)[0] + f"/{name}",
            admin_url.rsplit("/", 1)[0] + f"/{name}",
        )
    finally:
        await admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')
        await admin.close()


async def _tables(raw_url: str) -> tuple[set[str], str | None]:
    conn = await asyncpg.connect(raw_url)
    try:
        rows = await conn.fetch("select tablename from pg_tables where schemaname = 'public'")
        version = None
        if any(r["tablename"] == "alembic_version" for r in rows):
            version = await conn.fetchval("select version_num from alembic_version")
        return {r["tablename"] for r in rows}, version
    finally:
        await conn.close()


async def test_upgrade_on_a_fresh_database_actually_applies(scratch_db):
    url, raw = scratch_db
    _alembic(url, "upgrade", "head")
    tables, version = await _tables(raw)
    assert tables >= EXPECTED, f"migrations reported success but created {sorted(tables)}"
    head = _alembic(url, "heads").split()[0]
    assert version == head


async def test_full_downgrade_and_upgrade_cycle(scratch_db):
    url, raw = scratch_db
    _alembic(url, "upgrade", "head")
    _alembic(url, "downgrade", "base")
    tables, _ = await _tables(raw)
    assert not EXPECTED & tables, f"downgrade left {sorted(EXPECTED & tables)}"
    _alembic(url, "upgrade", "head")
    assert (await _tables(raw))[0] >= EXPECTED

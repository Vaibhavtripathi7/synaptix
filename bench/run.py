"""Per-replica capacity benchmark.

    uv run python -m bench.run            # 1 and 4 server processes, three paths, 15 s each

What it measures, and what it deliberately does not: the model is an external service whose
throughput is set by the provider and the caller's quota, not by this code, so it is out of
scope. The paths measured are the ones this system owns — the framework floor, an authenticated
database read, and a question answered from the cache (auth, per-workspace rate limit in Redis,
document versions, cache read, response). Those set how many replicas a given load needs.

Method: the server is pinned to cores 0-3 and the load generator (oha, in a container, so
nothing is installed on the host) to cores 4-7, so the client does not steal the server's CPU.
Production settings: JSON logs, no uvicorn access log. Rate limits are lifted: this measures
capacity, not policy (the limiter's own cost is still paid on every request)."""

import asyncio
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from redis.asyncio import Redis
from sqlalchemy import delete

from app.core.settings import get_settings
from app.db.models import DocStatus, Document
from app.db.session import SessionLocal, engine
from app.rag.cache import cache_key

PORT, DURATION, CONNECTIONS = 8010, "15s", 64
QUESTION = "What is the quality factor of a series RLC circuit?"
OUT = Path(__file__).with_name("results.md")


async def seed(token: str) -> str:
    """A ready document in a fresh workspace, and a cached answer for QUESTION about it — what a
    first, computed answer would have left behind."""
    settings = get_settings()
    workspace = hashlib.sha256(token.encode()).hexdigest()
    async with SessionLocal() as s:
        doc = Document(
            workspace_id=workspace,
            filename="bench.pdf",
            content_hash=secrets.token_hex(16),
            storage_key="x",
            status=DocStatus.ready,
            title="Bench",
        )
        s.add(doc)
        await s.commit()
        key = cache_key(workspace, {doc.id: doc.updated_at}, QUESTION, "standard", settings)
    passage = {
        "n": 1,
        "chunk_ids": [],
        "document_id": str(doc.id),
        "title": "Bench",
        "heading_path": ["Bench", "Resonance"],
        "page_start": 25,
        "page_end": 27,
        "kind": "text",
        "text": "Q = (1/R) sqrt(L/C). " * 40,
        "score": 9.0,
        "images": [],
    }
    payload = {
        "answer": "The quality factor is $Q = \\frac{1}{R}\\sqrt{L/C}$ [1]. " * 8,
        "passages": [passage] * 6,
        "queries": [QUESTION],
        "candidates": 40,
    }
    cache = Redis.from_url(settings.cache_redis_url)
    await cache.set(key, json.dumps(payload), ex=3600)
    await cache.aclose()
    await engine.dispose()  # pooled connections belong to this event loop; unseed runs in another
    return str(doc.id)


async def unseed() -> None:
    async with SessionLocal() as s:
        await s.execute(delete(Document).where(Document.filename == "bench.pdf"))
        await s.commit()
    await engine.dispose()


def serve(workers: int) -> subprocess.Popen:
    env = os.environ | {
        "SYNAPTIX_ENV": "prod",
        "SYNAPTIX_LOG_LEVEL": "WARNING",
        "SYNAPTIX_RATE_LIMITS": json.dumps({k: [1e9, 10**9] for k in ("model", "upload", "read")}),
        "SYNAPTIX_METRICS_PORT": "9110",
    }
    cmd = [
        "taskset",
        "-c",
        "0-3",
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--port",
        str(PORT),
        "--workers",
        str(workers),
        "--no-access-log",
    ]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/readyz", timeout=1)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("server did not become ready")


def load(path: str, token: str, body: str | None = None) -> dict:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--cpuset-cpus",
        "4-7",
        "ghcr.io/hatoo/oha:latest",
        "--no-tui",
        "--output-format",
        "json",
        "-z",
        DURATION,
        "-c",
        str(CONNECTIONS),
        "-H",
        f"x-workspace: {token}",
    ]
    if body:
        cmd += ["-m", "POST", "-T", "application/json", "-d", body]
    out = subprocess.run(cmd + [f"http://127.0.0.1:{PORT}{path}"], capture_output=True, text=True)
    r = json.loads(out.stdout)
    pct = r["latencyPercentiles"]
    return {
        "rps": r["summary"]["requestsPerSec"],
        "p50": pct["p50"] * 1000,
        "p99": pct["p99"] * 1000,
        "ok": r["summary"]["successRate"],
        "codes": r.get("statusCodeDistribution", {}),
    }


def main() -> None:
    token = secrets.token_urlsafe(32)
    doc = asyncio.run(seed(token))
    scenarios = [
        ("framework floor", "/healthz", None),
        ("authenticated DB read", "/api/v1/documents", None),
        (
            "cached answer",
            "/api/v1/query",
            json.dumps({"question": QUESTION, "document_ids": [doc]}),
        ),
    ]
    rows = []
    try:
        for workers in (1, 4):
            server = serve(workers)
            try:
                for name, path, body in scenarios:
                    load(path, token, body)  # warm-up: pools, caches and connections
                    r = load(path, token, body)
                    assert set(r["codes"]) == {"200"}, f"{name}: non-200 responses {r['codes']}"
                    rows.append((workers, name, r))
                    print(
                        f"{workers} proc  {name:24} {r['rps']:8.0f} rps  p50 {r['p50']:6.1f} ms  "
                        f"p99 {r['p99']:6.1f} ms  ok {r['ok']:.3f}  {r['codes']}",
                        flush=True,
                    )
            finally:
                server.terminate()
                server.wait()
    finally:
        asyncio.run(unseed())

    lines = [
        "| processes | path | req/s | p50 ms | p99 ms | success |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {w} | {n} | {r['rps']:.0f} | {r['p50']:.1f} | {r['p99']:.1f} | {r['ok']:.3f} |"
        for w, n, r in rows
    ]
    OUT.write_text(
        f"Server on 4 cores, load generator on 4 others, {CONNECTIONS} connections, "
        f"{DURATION} per run after a warm-up run.\n\n" + "\n".join(lines) + "\n"
    )
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()

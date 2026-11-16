import ipaddress
import random
import socket
import uuid

import pymupdf
import pytest
from sqlalchemy import delete, text

from app.core.settings import get_settings
from app.db.models import Chunk, DocStatus, Document, ElementKind, Section
from app.db.session import SessionLocal

DIM = get_settings().embedding_dim


@pytest.fixture(autouse=True)
def _generous_ip_limits(monkeypatch):
    """Every test client connects from 127.0.0.1, so per-IP buckets would make tests throttle
    each other. Lifted by default; the test that proves IP limits sets its own."""
    from app.core import limits

    for bucket in list(limits.IP_BUCKETS):
        monkeypatch.setitem(limits.IP_BUCKETS, bucket, (1e6, 10**6))


@pytest.fixture(autouse=True)
def _no_model_network(monkeypatch):
    """Tests must never reach a paid model API: a stray call spends someone's real quota.

    Two layers, because the first alone once let a request through - the SDK's HTTP stack
    (httpx -> httpcore -> anyio) IDNA-encodes the hostname, so it arrives here as bytes:
      1. resolving a Google API host fails the test, str or bytes;
      2. connecting anywhere but loopback or a private network fails the test, whatever the
         resolver, library or hostname. Tests only ever need the local Postgres and Redis."""
    resolve, connect, connect_ex = (
        socket.getaddrinfo,
        socket.socket.connect,
        socket.socket.connect_ex,
    )

    def guarded_resolve(host, *args, **kwargs):
        name = host.decode() if isinstance(host, bytes) else host
        if isinstance(name, str) and name.rstrip(".").endswith("googleapis.com"):
            raise RuntimeError(f"a test tried to reach {name}: model calls must be faked")
        return resolve(host, *args, **kwargs)

    def check(address) -> None:
        if isinstance(address, tuple) and address:
            ip = ipaddress.ip_address(address[0])
            if not (ip.is_loopback or ip.is_private):
                raise RuntimeError(f"a test tried to connect to {ip}: model calls must be faked")

    def guarded_connect(sock, address):
        check(address)
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        check(address)
        return connect_ex(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_resolve)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


WORKSPACE = "a" * 64  # a workspace id (already hashed) for fixtures that bypass the API


def s3_settings(**overrides):
    """Settings for the S3 test server in docker-compose.yml (`--profile s3`), fresh bucket."""
    from pydantic import SecretStr

    values = {
        "storage_backend": "s3",
        "s3_endpoint_url": "http://localhost:9005",
        "s3_bucket": f"synaptix-test-{uuid.uuid4().hex[:6]}",
        "s3_access_key": "synaptix",
        "s3_secret_key": SecretStr("synaptix-dev-secret"),
    }
    return get_settings().model_copy(update=values | overrides)


def vec(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(DIM)]


@pytest.fixture
async def db():
    """Skips the test when Postgres is not reachable, so unit tests still run anywhere."""
    try:
        async with SessionLocal() as s:
            await s.execute(text("select 1"))
    except Exception:
        pytest.skip("database not reachable")


@pytest.fixture
async def seeded(db):
    """One ready document: a short section holding three chunks with known text and vectors."""
    doc = Document(
        workspace_id=WORKSPACE,
        filename="t.pdf",
        content_hash=uuid.uuid4().hex,
        storage_key="x",
        title="Test Doc",
        status=DocStatus.ready,
        page_count=2,
    )
    sec = Section(
        order_idx=0,
        heading_path=["Test Doc", "Methods"],
        page_start=1,
        page_end=2,
        content="Short section body about isochrones.",
    )
    texts = [
        "The isochrone method has drawbacks: limited variables and constraints.",
        "Genetic algorithms approximate the Pareto front for multi-objective routing.",
        "Dijkstra computes shortest paths on a grid graph.",
    ]
    async with SessionLocal() as s:
        s.add(doc)
        await s.flush()
        sec.document_id = doc.id
        s.add(sec)
        await s.flush()
        s.add_all(
            Chunk(
                document_id=doc.id,
                section_id=sec.id,
                order_idx=i,
                kind=ElementKind.text,
                content=t,
                token_count=12,
                page_start=1,
                page_end=1,
                embedding=vec(i),
            )
            for i, t in enumerate(texts)
        )
        await s.commit()
        yield doc.id
        # A statement, not session.delete: the test may have deleted the document already.
        await s.execute(delete(Document).where(Document.id == doc.id))
        await s.commit()


@pytest.fixture(scope="session")
def small_pdf() -> bytes:
    """Three-page PDF: title, sections with prose, a figure, and a scanned (image-only) page."""
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 90), "Quarterly Performance Review", fontsize=22)
    p.insert_text((72, 140), "1. Revenue", fontsize=16)
    y = 170
    for i in range(12):
        p.insert_text(
            (72, y),
            f"Revenue grew {i + 3}% in region {i} driven by new enterprise accounts.",
            fontsize=10,
        )
        y += 16
    p.insert_text((72, y + 20), "2. Costs", fontsize=16)
    p.insert_text(
        (72, y + 50),
        "Operating costs were flat year over year despite headcount growth.",
        fontsize=10,
    )
    p.draw_rect(
        pymupdf.Rect(72, y + 80, 372, y + 280), color=(0, 0, 1), fill=(0.8, 0.8, 1), width=2
    )
    p.insert_text((80, y + 300), "Figure 1. Cost breakdown by department.", fontsize=9)

    p2 = doc.new_page()
    p2.insert_text((72, 90), "3. Outlook", fontsize=16)
    p2.insert_text(
        (72, 120), "Guidance for next year assumes continued enterprise momentum.", fontsize=10
    )

    scan = doc.new_page()  # image-only page
    pix = doc[0].get_pixmap(dpi=50)  # `p` is stale once more pages exist
    scan.insert_image(scan.rect, pixmap=pix)
    return doc.tobytes()


@pytest.fixture
async def redis():
    from arq import create_pool
    from arq.connections import RedisSettings

    try:
        pool = await create_pool(RedisSettings.from_dsn(get_settings().redis_url), retry=0)
        await pool.ping()
    except Exception:
        pytest.skip("redis not reachable")
    yield pool
    await pool.aclose()


@pytest.fixture
def fake_gemini(monkeypatch):
    """Every Gemini client built during the test talks to a FakeGemini, through the real SDK."""
    import httpx
    from google import genai

    from app.llm import client as client_module
    from app.rag import answer as answer_module
    from tests.fake_gemini import FakeGemini

    fake = FakeGemini(DIM)
    real = genai.Client

    def client(**kwargs):
        transport = {"async_client_args": {"transport": httpx.MockTransport(fake.handle)}}
        return real(
            **kwargs | {"http_options": kwargs["http_options"].model_copy(update=transport)}
        )

    monkeypatch.setattr(client_module.genai, "Client", client)
    answer_module.answerer_for.cache_clear()  # stacks cached before the patch would bypass it
    yield fake
    answer_module.answerer_for.cache_clear()

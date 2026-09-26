from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Every field can be set as SYNAPTIX_<FIELD> or in .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SYNAPTIX_", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # Infra (host ports are non-default on purpose so local services never collide)
    database_url: str = "postgresql+asyncpg://synaptix:synaptix@localhost:5433/synaptix"
    redis_url: str = "redis://localhost:6380/0"  # job queue: must never evict
    cache_redis_url: str = "redis://localhost:6381/0"  # answer cache: allkeys-lru, may evict
    answer_cache_ttl_s: int = 86_400  # 0 disables the cache
    storage_dir: Path = Path("data/storage")  # source PDFs and extracted figures
    # Workspaces are anonymous, so these caps are what bound one visitor's disk and CPU use.
    max_upload_mb: int = 100
    max_pages: int = 500
    max_documents_per_workspace: int = 50
    # Comma-separated bearer keys accepted on /api/*. Empty (the default) leaves the API open,
    # which is only appropriate for local development.
    api_keys: str = ""
    metrics_port: int = 9100  # API process; internal only
    worker_metrics_port: int = 9101

    # Gemini: "vertex" uses ADC + project, "api_key" uses an AI Studio key
    gemini_backend: Literal["vertex", "api_key"] = "api_key"
    google_api_key: SecretStr | None = None
    gcp_project: str | None = None
    gcp_location: str = "us-central1"
    llm_model: str = "gemini-3.5-flash"  # fast: enrichment, rerank, Q&A
    llm_model_strong: str = "gemini-pro-latest"  # reasoning: analysis agents
    # Tried in order when the primary returns 429/503 (quota, outage). JSON list in env.
    llm_fallback_models: list[str] = ["gemini-3-flash-preview", "gemini-3.5-flash-lite"]
    llm_concurrency: int = 4
    # Workers wait out a per-minute rate limit; API processes should not sleep ~30s inside a
    # request, so they set this false and fail over at once.
    llm_wait_on_rate_limit: bool = True
    max_inflight_model_requests: int = 64  # per replica; above it, 503 + Retry-After
    # Per-workspace token buckets: (sustained per second, burst). Model-backed work is scarce,
    # ingestion is the most expensive thing a caller can trigger, reads are cheap.
    rate_limits: dict[str, tuple[float, int]] = {
        "model": (0.5, 10),
        "upload": (0.2, 5),
        "read": (20.0, 100),
    }
    # USD per million tokens, (input, output), keyed by model. Empty: usage is reported in
    # tokens only — prices change and a stale default would report a confident wrong number.
    llm_prices_per_mtok: dict[str, tuple[float, float]] = {}

    # Retrieval stack. `local` variants need the optional `sentence-transformers` extra.
    embedding_provider: Literal["gemini", "local"] = "gemini"
    embedding_model: str = "gemini-embedding-001"
    local_embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024  # bge-m3 native; gemini-embedding-001 truncated (Matryoshka)
    reranker: Literal["llm", "local", "none"] = "llm"
    local_reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Ingestion
    ocr_batch_pages: int = 6  # scanned pages per vision request; the main cost lever

    # Chunking
    chunk_tokens: int = 400
    chunk_overlap_tokens: int = 60

    # Retrieval
    query_expansion: bool = True  # one extra LLM call: paraphrased queries improve recall
    retrieval_k: int = 40  # fused candidates per question
    rerank_top_n: int = 8  # passages that survive reranking
    context_tokens: int = 6000  # budget for the answer prompt


@lru_cache
def get_settings() -> Settings:
    return Settings()

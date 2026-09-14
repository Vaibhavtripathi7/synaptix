from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Every field can be set as SYNAPTIX_<FIELD> or in .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SYNAPTIX_", extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # Infra (host ports are non-default on purpose so local services never collide)
    database_url: str = "postgresql+asyncpg://synaptix:synaptix@localhost:5433/synaptix"

    # Vector width, fixed by the schema: gemini-embedding-001 truncated to 1024, or bge-m3 native.
    embedding_dim: int = 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()

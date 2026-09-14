"""Embedding providers behind one interface. `gemini` is managed; `local` runs bge-m3 in-process."""

import asyncio
from typing import Protocol

from app.core.settings import Settings
from app.llm.client import Gemini


class Embedder(Protocol):
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


class GeminiEmbedder:
    def __init__(self, llm: Gemini) -> None:
        self._llm = llm

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._llm.embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._llm.embed([text], query=True))[0]


class LocalEmbedder:
    """sentence-transformers model (default bge-m3, 1024-d). Needs `uv sync --extra local`."""

    def __init__(self, model_name: str, dim: int) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, truncate_dim=dim)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = await asyncio.to_thread(
            self._model.encode, texts, batch_size=16, normalize_embeddings=True
        )
        return vectors.tolist()

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


def build_embedder(settings: Settings, llm: Gemini) -> Embedder:
    if settings.embedding_provider == "local":
        return LocalEmbedder(settings.local_embedding_model, settings.embedding_dim)
    return GeminiEmbedder(llm)

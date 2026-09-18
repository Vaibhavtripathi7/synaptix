"""Async wrapper over google-genai: one place for auth, retries, concurrency, structured output."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import partial
from typing import TypeVar

import structlog
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from app.core.settings import Settings

T = TypeVar("T", bound=BaseModel)
Content = str | list[str | types.Part]
log = structlog.get_logger()

_FAILOVER_CODES = {429, 503}  # quota exhausted / model overloaded: try the next model


class Gemini:
    def __init__(self, settings: Settings) -> None:
        http = types.HttpOptions(
            timeout=120_000,
            # Transient transport errors are retried in place; 429/503 fail over instead.
            retry_options=types.HttpRetryOptions(
                attempts=3, initial_delay=1, max_delay=8, http_status_codes=[408, 500, 502, 504]
            ),
        )
        if settings.gemini_backend == "vertex":
            self._client = genai.Client(
                vertexai=True,
                project=settings.gcp_project,
                location=settings.gcp_location,
                http_options=http,
            )
        else:
            if settings.google_api_key is None:
                raise RuntimeError("SYNAPTIX_GOOGLE_API_KEY is required for the api_key backend")
            self._client = genai.Client(
                api_key=settings.google_api_key.get_secret_value(), http_options=http
            )
        self._chains = {
            False: [settings.llm_model, *settings.llm_fallback_models],
            True: [settings.llm_model_strong, *settings.llm_fallback_models],
        }
        self._embedding_model = settings.embedding_model
        self._embedding_dim = settings.embedding_dim
        self._sem = asyncio.Semaphore(settings.llm_concurrency)

    @staticmethod
    def image(png: bytes) -> types.Part:
        return types.Part.from_bytes(data=png, mime_type="image/png")

    def _config(
        self,
        system: str | None,
        temperature: float,
        max_tokens: int | None,
        schema: type[BaseModel] | None,
        think: bool,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if schema else None,
            response_schema=schema,
            # Models think by default; extraction-style calls run at the lowest thinking level.
            thinking_config=None if think else types.ThinkingConfig(thinking_level="minimal"),
        )

    async def _call(self, chain: list[str], call: Callable[[str], Awaitable[T]]) -> T:
        """Runs `call(model)` down the chain: a quota error or an overloaded model fails over to
        the next entry; anything else, or the last entry failing, raises."""
        for i, model in enumerate(chain):
            try:
                return await call(model)
            except errors.APIError as exc:
                if exc.code not in _FAILOVER_CODES or i == len(chain) - 1:
                    raise
                log.warning("llm.failover", model=model, code=exc.code, next=chain[i + 1])
        raise AssertionError("unreachable")

    async def _generate(self, content: Content, config, strong: bool):
        async with self._sem:
            return await self._call(
                self._chains[strong],
                lambda m: self._client.aio.models.generate_content(
                    model=m, contents=content, config=config
                ),
            )

    async def generate(
        self,
        content: Content,
        *,
        system: str | None = None,
        strong: bool = False,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        think: bool = False,
    ) -> str:
        config = self._config(system, temperature, max_tokens, None, think)
        resp = await self._generate(content, config, strong)
        return resp.text or ""

    async def generate_structured(
        self,
        content: Content,
        schema: type[T],
        *,
        system: str | None = None,
        strong: bool = False,
        temperature: float = 0.1,
        think: bool = False,
    ) -> T:
        config = self._config(system, temperature, None, schema, think)
        resp = await self._generate(content, config, strong)
        parsed = resp.parsed
        if isinstance(parsed, schema):
            return parsed
        return schema.model_validate_json(resp.text or "{}")

    async def stream(
        self,
        content: Content,
        *,
        system: str | None = None,
        strong: bool = False,
        temperature: float = 0.2,
        think: bool = True,
    ) -> AsyncIterator[str]:
        config = self._config(system, temperature, None, None, think)

        async def open_stream(model: str):
            it = await self._client.aio.models.generate_content_stream(
                model=model, contents=content, config=config
            )
            # Quota/overload errors surface on the first read, so read it inside the failover.
            return it, await anext(it, None)

        async with self._sem:
            it, chunk = await self._call(self._chains[strong], open_stream)
            while chunk is not None:
                if chunk.text:
                    yield chunk.text
                chunk = await anext(it, None)

    async def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """Embeds in batches of 64 (API cap is 100). Truncates to the configured dimension
        (Matryoshka), which is why one column width serves both providers."""
        cfg = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
            output_dimensionality=self._embedding_dim,
        )
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            async with self._sem:
                resp = await self._call(
                    [self._embedding_model], partial(self._embed_batch, batch, cfg)
                )
            out.extend(e.values for e in resp.embeddings)
        return out

    def _embed_batch(self, batch: list[str], cfg: types.EmbedContentConfig, model: str):
        return self._client.aio.models.embed_content(model=model, contents=batch, config=cfg)

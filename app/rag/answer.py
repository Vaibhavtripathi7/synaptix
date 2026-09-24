"""Question answering over ingested documents.

    question ─► (expand) ─► embed ─► hybrid search ─► rerank ─► context ─► Gemini ─► tokens
                                                                            └─► citations

Citations are `[n]` markers that refer to the numbered passages the model was shown; the
passages are returned alongside the answer so a client can resolve them to page/element."""

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import Settings, get_settings
from app.db.models import Document
from app.llm.client import Gemini
from app.llm.embeddings import Embedder, build_embedder
from app.retrieval.context import Passage, build_context, render
from app.retrieval.hybrid import Hit, hybrid_search
from app.retrieval.rerank import Reranker, build_reranker

_EXPAND_SYSTEM = (
    "Rewrite the user's question as two alternative search queries that use different "
    "wording and likely document vocabulary (technical terms, synonyms, the underlying "
    "entities). Stay within the subject matter of the documents being searched. "
    "Do not answer the question."
)
_ANSWER_SYSTEM = (
    "You answer questions using only the numbered passages provided. Every factual claim "
    "must end with the citation(s) it comes from, like [2] or [1][3]. Prefer exact figures, "
    "names and units from the passages. If the passages do not contain the answer, say what "
    "is missing instead of guessing. Answer in Markdown, with mathematics as LaTeX between "
    "$ delimiters."
)

Style = Literal["brief", "standard", "explain"]
_STYLE: dict[Style, str] = {
    "brief": " Answer in one or two sentences, or a single formula. No preamble, no restating "
    "the question, no elaboration beyond what was asked.",
    "standard": " Be direct and complete: cover the answer and the reasoning behind it without "
    "padding.",
    "explain": " Teach the answer to someone meeting it for the first time: define the terms "
    "you use, work through the reasoning step by step, show intermediate steps for any "
    "derivation, and end with the intuition for why the result holds.",
}


class _Queries(BaseModel):
    queries: list[str]


@dataclass(slots=True)
class Retrieval:
    passages: list[Passage]
    queries: list[str]
    candidates: int


class Answerer:
    def __init__(
        self, llm: Gemini, embedder: Embedder, reranker: Reranker, settings: Settings
    ) -> None:
        self.llm, self.embedder, self._reranker, self.settings = llm, embedder, reranker, settings

    async def search(
        self,
        session: AsyncSession,
        question: str,
        document_ids: list[uuid.UUID],
        *,
        expand: bool,
        rerank: bool,
        top_n: int,
    ) -> tuple[list[Hit], list[str], int]:
        """Retrieval only: (top hits, query variants used, fused candidate count)."""
        queries = [question]
        if expand:
            try:
                titles = await self._titles(session, document_ids)
                prompt = f"Documents being searched: {'; '.join(titles)}\n\nQuestion: {question}"
                extra = await self.llm.generate_structured(prompt, _Queries, system=_EXPAND_SYSTEM)
                queries += [q for q in extra.queries[:2] if q and q != question]
            except Exception:  # expansion is an optimisation, never a dependency
                pass
        vectors = (
            await self.embedder.embed_documents(queries)
            if len(queries) > 1
            else [await self.embedder.embed_query(question)]
        )
        hits = await hybrid_search(
            session,
            list(zip(queries, vectors, strict=True)),
            document_ids=document_ids,
            k=self.settings.retrieval_k,
        )
        candidates = len(hits)
        if rerank:
            hits = await self._reranker.rerank(question, hits, top_n)
        return hits[:top_n], queries, candidates

    async def retrieve(
        self, session: AsyncSession, question: str, document_ids: list[uuid.UUID]
    ) -> Retrieval:
        hits, queries, candidates = await self.search(
            session,
            question,
            document_ids,
            expand=self.settings.query_expansion,
            rerank=True,
            top_n=self.settings.rerank_top_n,
        )
        return Retrieval(build_context(hits, self.settings.context_tokens), queries, candidates)

    @staticmethod
    async def _titles(session: AsyncSession, document_ids: list[uuid.UUID]) -> list[str]:
        stmt = select(Document.title).where(Document.id.in_(document_ids)).limit(20)
        return [t for t in await session.scalars(stmt) if t]

    async def stream(
        self, question: str, passages: list[Passage], style: Style = "standard"
    ) -> AsyncIterator[str]:
        prompt = f"Passages:\n\n{render(passages)}\n\nQuestion: {question}"
        system = _ANSWER_SYSTEM + _STYLE[style]
        async for token in self.llm.stream(prompt, system=system):
            yield token

    async def answer(
        self, question: str, passages: list[Passage], style: Style = "standard"
    ) -> str:
        return "".join([t async for t in self.stream(question, passages, style)])


class MissingKeyError(RuntimeError):
    """No user key was supplied and this server does not provide one."""


@lru_cache(maxsize=64)
def answerer_for(api_key: str | None) -> Answerer:
    """The client stack for one Gemini key, reused across requests and jobs.

    Users bring their own key, so each key gets its own client and therefore its own
    concurrency limit: one user's burst cannot starve another's. With no user key the server's
    configured key is used, if the operator set one; a hosted deployment simply doesn't."""
    settings = get_settings()
    if api_key:
        settings = settings.model_copy(
            update={"google_api_key": SecretStr(api_key), "gemini_backend": "api_key"}
        )
    elif settings.gemini_backend == "api_key" and settings.google_api_key is None:
        raise MissingKeyError("add a Gemini API key: this server does not provide one")
    llm = Gemini(settings)
    return Answerer(llm, build_embedder(settings, llm), build_reranker(settings, llm), settings)

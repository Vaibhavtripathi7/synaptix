"""Second-stage reranking of fused candidates. `llm` is a listwise Gemini scorer (no extra
dependencies), `local` is a cross-encoder (bge-reranker-v2-m3), `none` keeps the RRF order."""

import asyncio
from functools import cache
from typing import Protocol

from pydantic import BaseModel, Field

from app.core.settings import Settings
from app.llm.client import Gemini
from app.retrieval.hybrid import Hit

_PASSAGE_CHARS = 1200  # enough to judge relevance; keeps the listwise prompt small


class Reranker(Protocol):
    async def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...


class NoReranker:
    async def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        return hits[:top_n]


class _Score(BaseModel):
    index: int
    relevance: int = Field(ge=0, le=10)


class _Scores(BaseModel):
    scores: list[_Score]


class LLMReranker:
    """Listwise: the model sees all candidates at once and grades each 0-10 for whether it
    contains information that answers the query. One call per rerank."""

    _SYSTEM = (
        "You grade search passages for a question-answering system. For each passage, output "
        "how well it helps answer the query: 10 = directly answers it, 5 = relevant background, "
        "0 = unrelated. Judge only the passage text. Grade every passage exactly once."
    )

    def __init__(self, llm: Gemini, passage_chars: int = _PASSAGE_CHARS) -> None:
        self._llm, self._chars = llm, passage_chars

    async def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if len(hits) <= 1:
            return hits[:top_n]
        listing = "\n\n".join(f"[{i}] {h.chunk.content[: self._chars]}" for i, h in enumerate(hits))
        result = await self._llm.generate_structured(
            f"Query: {query}\n\nPassages:\n\n{listing}", _Scores, system=self._SYSTEM
        )
        graded = {s.index: s.relevance for s in result.scores if 0 <= s.index < len(hits)}
        for i, h in enumerate(hits):
            h.rerank_score = float(graded.get(i, 0))
        # Tie-break on the fused score so ungraded passages keep their retrieval order.
        return sorted(hits, key=lambda h: (h.rerank_score, h.score), reverse=True)[:top_n]


class LocalReranker:
    """Cross-encoder scoring in-process. Needs `uv sync --extra local`."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    async def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        pairs = [(query, h.chunk.content[:_PASSAGE_CHARS]) for h in hits]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        for h, s in zip(hits, scores, strict=True):
            h.rerank_score = float(s)
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:top_n]


@cache
def _local_reranker(model_name: str) -> LocalReranker:
    return LocalReranker(model_name)  # one copy per process, shared by every user's stack


def build_reranker(settings: Settings, llm: Gemini) -> Reranker:
    match settings.reranker:
        case "local":
            return _local_reranker(settings.local_reranker_model)
        case "llm":
            return LLMReranker(llm)
        case _:
            return NoReranker()

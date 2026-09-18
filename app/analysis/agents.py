"""Retrieval-backed analysis agents.

Each agent is a set of retrieval *facets* — short descriptions of what it needs to see — plus a
structured prompt. Facet hits are unioned into one budgeted, numbered context, so the model sees
the relevant parts of a document rather than a truncated dump of it, and the output is a
Pydantic model whose items cite the passages they rest on."""

import asyncio
import uuid
from dataclasses import dataclass

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.schemas import Comparison, CompetitorProfile, KeyAnalysis, Strategy, Swot
from app.rag.answer import Answerer
from app.retrieval.context import Passage, build_context, render
from app.retrieval.hybrid import Hit, hybrid_search

_HITS_PER_FACET = 6
_CONTEXT_TOKENS = 9000
_CITE_RULE = (
    " Work only from the numbered passages and list the passage numbers you used in each "
    "`evidence`. If the passages do not cover a category, return an empty list for it rather "
    "than inventing content."
)


@dataclass(frozen=True, slots=True)
class Agent:
    name: str
    system: str
    facets: tuple[str, ...]
    schema: type[BaseModel]


PROFILE = Agent(
    "profile",
    "You are a competitor profiling analyst. Build a profile of the company the passages "
    "describe: basic information, products and services, market presence and sales. Keep "
    "concrete names, figures and dates." + _CITE_RULE,
    (
        "company overview, history, headquarters and industry",
        "products, services, features and pricing",
        "customers, target market, market share and geographic reach",
        "revenue, sales channels and financial results",
    ),
    CompetitorProfile,
)

SWOT = Agent(
    "swot",
    "You are a strategy analyst. Perform a SWOT analysis of the company the passages describe: "
    "strengths, weaknesses, opportunities and threats, each grounded in the passages." + _CITE_RULE,
    (
        "competitive advantages, differentiators and capabilities",
        "risks, weaknesses, challenges and limitations",
        "growth plans, new markets and opportunities",
        "competitors, regulation and market threats",
    ),
    Swot,
)

STRATEGY = Agent(
    "strategy",
    "You are a competitive strategy manager. From the passages, describe the company's likely "
    "future strategy as specific actions, analyse its impact on other players in the market, "
    "and propose concrete counter-strategies." + _CITE_RULE,
    (
        "strategy, plans, roadmap and future direction",
        "investments, partnerships and acquisitions",
        "marketing, pricing and go-to-market approach",
    ),
    Strategy,
)

KEY_ANALYSIS = Agent(
    "key_analysis",
    "You are a competitive analyst. Combine the profile, SWOT analysis and strategy already "
    "produced, together with the passages, into a key analysis: an executive summary, the "
    "company's main competitive advantages and disadvantages, and actionable recommendations."
    + _CITE_RULE,
    (
        "key results, metrics and performance figures",
        "competitive position and differentiation",
    ),
    KeyAnalysis,
)

COMPARISON_SYSTEM = (
    "Compare the two documents below. Identify their key differences and similarities in "
    "content and approach, then what the first company can learn from the second and what the "
    "second can learn from the first. Be specific and actionable." + _CITE_RULE
)
COMPARISON_FACETS = (
    "company overview, products and services",
    "strategy, plans and market approach",
    "results, metrics and financial performance",
    "strengths, weaknesses and risks",
)


async def gather_context(
    answerer: Answerer,
    session: AsyncSession,
    facets: tuple[str, ...],
    document_ids: list[uuid.UUID],
    *,
    start_at: int = 1,
) -> list[Passage]:
    """Retrieves for every facet concurrently and packs the union into one numbered context.
    No query expansion and no reranker: a facet is already a single intent, and RRF order is
    good enough for a union."""

    async def one(facet: str) -> list[Hit]:
        vector = await answerer.embedder.embed_query(facet)
        return await hybrid_search(
            session, [(facet, vector)], document_ids=document_ids, k=answerer.settings.retrieval_k
        )

    best: dict[uuid.UUID, Hit] = {}
    for hits in await asyncio.gather(*(one(f) for f in facets)):
        for h in hits[:_HITS_PER_FACET]:
            if h.chunk.id not in best or h.score > best[h.chunk.id].score:
                best[h.chunk.id] = h
    ranked = sorted(best.values(), key=lambda h: h.score, reverse=True)
    passages = build_context(ranked, _CONTEXT_TOKENS)
    for p in passages:
        p.n += start_at - 1
    return passages


async def run_agent(
    agent: Agent,
    answerer: Answerer,
    session: AsyncSession,
    document_ids: list[uuid.UUID],
    *,
    inputs: dict[str, BaseModel] | None = None,
) -> tuple[BaseModel, list[Passage]]:
    passages = await gather_context(answerer, session, agent.facets, document_ids)
    result = await answerer.llm.generate_structured(
        _prompt(passages, inputs),
        agent.schema,
        system=agent.system,
        strong=True,
        think=True,
    )
    return result, passages


async def compare(
    answerer: Answerer, session: AsyncSession, first: uuid.UUID, second: uuid.UUID
) -> tuple[Comparison, list[Passage]]:
    a = await gather_context(answerer, session, COMPARISON_FACETS, [first])
    b = await gather_context(answerer, session, COMPARISON_FACETS, [second], start_at=len(a) + 1)
    prompt = f"First document passages:\n\n{render(a)}\n\nSecond document passages:\n\n{render(b)}"
    result = await answerer.llm.generate_structured(
        prompt, Comparison, system=COMPARISON_SYSTEM, strong=True, think=True
    )
    return result, a + b


def _prompt(passages: list[Passage], inputs: dict[str, BaseModel] | None) -> str:
    parts = [
        f"{name}:\n{model.model_dump_json(indent=1)}" for name, model in (inputs or {}).items()
    ]
    parts.append(f"Passages:\n\n{render(passages)}")
    return "\n\n".join(parts)

"""Retrieval-backed analysis agents.

Two shapes of agent, because two shapes of question:

* **Retrieval** (`run_agent`) — for outputs answering a question about the material. Facet hits
  are unioned into one budgeted, numbered context, so the model sees the *relevant* parts.
* **Coverage** (`run_mapped_agent`) — for outputs that must not miss anything, like a study
  guide. Retrieval is the wrong primitive there: top-k by definition drops the tail. Instead
  every chunk of the document is numbered once and mapped over in batches, and the fragments
  are merged. Passage numbers stay global across batches, so `evidence` still resolves.

Either way the output is a Pydantic model whose items cite the passages they rest on."""

import asyncio
import uuid
from dataclasses import dataclass

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.analysis.schemas import PaperComparison, PracticeSet, StudyOutline
from app.db.models import Chunk
from app.ingest.chunk import count_tokens
from app.rag.answer import Answerer
from app.retrieval.context import Passage, build_context, render
from app.retrieval.hybrid import Hit, hybrid_search

_HITS_PER_FACET = 6
_CONTEXT_TOKENS = 9000
_CITE_RULE = (
    " Work only from the numbered passages and list the passage numbers you used in each "
    "`evidence`. If the passages do not cover a category, return an empty list for it rather "
    "than inventing content. Write every symbol, subscript and expression as LaTeX between "
    "$ delimiters — $B_L$, not B_L — so it renders."
)


@dataclass(frozen=True, slots=True)
class Agent:
    name: str
    system: str
    facets: tuple[str, ...]
    schema: type[BaseModel]


OUTLINE = Agent(
    "outline",
    "You are an experienced tutor building revision material from a student's source documents "
    "(lecture notes, slides or a paper). Be exhaustive about the passages you are given: "
    "capture every concept a student must understand, every formula that appears (with each "
    "symbol and its units defined), every worked example, and the conditions, assumptions and "
    "mistakes that trip people up. Use the document's own notation and numbers, and do not "
    "summarise several formulas into one." + _CITE_RULE,
    (
        "definitions, key concepts and terminology",
        "equations, formulas and mathematical relationships",
        "worked examples, solved problems and numerical calculations",
        "derivations, proofs and explanations of why a method works",
        "assumptions, conditions, special cases and common mistakes",
    ),
    StudyOutline,
)

PRACTICE = Agent(
    "practice",
    "You are setting practice questions on the material below. Using the outline already "
    "extracted and the passages, write at least eight questions spread across recall, "
    "application and analysis, covering different parts of the outline rather than one topic, "
    "and answer each one. Base every question on material actually present; where the document "
    "gives numbers, use them." + _CITE_RULE,
    (
        "worked examples, solved problems and numerical calculations",
        "equations, formulas and mathematical relationships",
        "definitions, key concepts and terminology",
    ),
    PracticeSet,
)

COMPARISON_SYSTEM = (
    "Compare these two research documents for someone deciding what to read, cite or build on. "
    "Cover the problem each addresses, their methods, their data and experimental setup, their "
    "results, and their strengths and limitations; finish with how they relate and when to "
    "prefer each. Be specific: keep method names, dataset names and reported numbers." + _CITE_RULE
)
COMPARISON_FACETS = (
    "the problem being addressed, motivation and contributions",
    "method, model architecture and algorithm",
    "datasets, benchmarks, experimental setup and metrics",
    "results, performance numbers and comparisons",
    "limitations, assumptions and future work",
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


async def document_passages(session: AsyncSession, document_ids: list[uuid.UUID]) -> list[Passage]:
    """The whole document as numbered passages, in reading order.

    Chunks rather than sections: a section can be the entire document (a scanned deck with no
    detected headings is one section), and an `evidence` number that resolves to "the document"
    is worthless. A chunk always carries its own page range and id, so every citation lands on
    a page the reader can check."""
    chunks = list(
        await session.scalars(
            select(Chunk)
            .options(selectinload(Chunk.section), selectinload(Chunk.document))
            .where(Chunk.document_id.in_(document_ids))
            .order_by(Chunk.document_id, Chunk.order_idx)
        )
    )
    return [
        Passage(
            n=i + 1,
            chunk_ids=[c.id],
            document_id=c.document_id,
            title=c.document.title or c.document.filename,
            heading_path=c.section.heading_path if c.section else [],
            page_start=c.page_start,
            page_end=c.page_end,
            kind=c.kind,
            text=c.content,
            score=0.0,
        )
        for i, c in enumerate(chunks)
    ]


def _batch(passages: list[Passage], budget: int) -> list[list[Passage]]:
    batches: list[list[Passage]] = [[]]
    spent = 0
    for p in passages:
        cost = count_tokens(p.text)
        if spent + cost > budget and batches[-1]:
            batches.append([])
            spent = 0
        batches[-1].append(p)
        spent += cost
    return [b for b in batches if b]


async def run_mapped_agent(
    agent: Agent,
    answerer: Answerer,
    session: AsyncSession,
    document_ids: list[uuid.UUID],
    *,
    inputs: dict[str, BaseModel] | None = None,
) -> tuple[BaseModel, list[Passage]]:
    """Maps the agent over the whole document in batches and merges the fragments."""
    passages = await document_passages(session, document_ids)
    if not passages:
        return agent.schema.model_construct(), []
    batches = _batch(passages, _CONTEXT_TOKENS)

    async def one(batch: list[Passage]) -> BaseModel:
        return await answerer.llm.generate_structured(
            _prompt(batch, inputs),
            agent.schema,
            system=agent.system,
            strong=True,
            think=True,
        )

    fragments = await asyncio.gather(*(one(b) for b in batches))
    return _merge(fragments), passages


def _merge(fragments: list[BaseModel]) -> BaseModel:
    """Concatenates list fields across fragments, de-duplicating by the item's name or text;
    scalar fields take the longest non-empty value."""
    merged: dict = {}
    for field in type(fragments[0]).model_fields:
        values = [getattr(f, field) for f in fragments]
        if not isinstance(values[0], list):
            merged[field] = max((v for v in values if v), key=len, default="")
            continue
        seen: dict[str, object] = {}
        for item in (i for v in values for i in v):
            key = (
                (
                    getattr(item, "name", None)
                    or getattr(item, "question", None)
                    or getattr(item, "text", "")
                )
                .strip()
                .casefold()
            )
            if key and key not in seen:
                seen[key] = item
        merged[field] = list(seen.values())
    return type(fragments[0])(**merged)


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
) -> tuple[PaperComparison, list[Passage]]:
    a = await gather_context(answerer, session, COMPARISON_FACETS, [first])
    b = await gather_context(answerer, session, COMPARISON_FACETS, [second], start_at=len(a) + 1)
    prompt = f"First document passages:\n\n{render(a)}\n\nSecond document passages:\n\n{render(b)}"
    result = await answerer.llm.generate_structured(
        prompt, PaperComparison, system=COMPARISON_SYSTEM, strong=True, think=True
    )
    return result, a + b


def _prompt(passages: list[Passage], inputs: dict[str, BaseModel] | None) -> str:
    parts = [
        f"{name}:\n{model.model_dump_json(indent=1)}" for name, model in (inputs or {}).items()
    ]
    parts.append(f"Passages:\n\n{render(passages)}")
    return "\n\n".join(parts)

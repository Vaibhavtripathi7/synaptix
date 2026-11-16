"""Entity-relation graph of a document, extracted per batch of sections and merged.
Rendered client-side; also a compact map of what a document is about."""

import asyncio

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, Section
from app.db.session import end_read
from app.ingest.chunk import count_tokens
from app.llm.client import Gemini
from app.llm.untrusted import Fence

_BATCH_TOKENS = 12_000
_SYSTEM = (
    "Extract the knowledge graph of the text: the most important named entities (organisations, "
    "people, products, technologies, methods, metrics, locations, concepts) and the relations "
    "between them as (source, predicate, target) with a short verb-phrase predicate. Use the "
    "entity names exactly as they appear; merge obvious aliases. At most 40 entities and 60 "
    "relations per request; prefer relations stated explicitly in the text. `section` is the "
    "number of the section the relation comes from."
)


class Entity(BaseModel):
    name: str
    type: str = Field(
        description="organisation|person|product|technology|method|metric|location|concept"
    )


class Relation(BaseModel):
    source: str
    target: str
    predicate: str
    section: int


class Graph(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


def _extract(llm: Gemini, text: str):
    fence = Fence()
    return llm.generate_structured(fence.wrap(text, "sections"), Graph, system=_SYSTEM + fence.rule)


async def build_graph(session: AsyncSession, doc: Document, llm: Gemini) -> dict:
    sections = list(
        await session.scalars(
            select(Section).where(Section.document_id == doc.id).order_by(Section.order_idx)
        )
    )
    await end_read(session)
    batches: list[list[str]] = [[]]
    spent = 0
    for s in sections:
        text = f"### Section {s.order_idx}: {' > '.join(s.heading_path)}\n{s.content}"
        cost = count_tokens(text)
        if spent + cost > _BATCH_TOKENS and batches[-1]:
            batches.append([])
            spent = 0
        batches[-1].append(text)
        spent += cost

    parts = await asyncio.gather(*(_extract(llm, "\n\n".join(b)) for b in batches if b))
    pages = {s.order_idx: (s.page_start, s.page_end) for s in sections}
    nodes: dict[str, dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}
    for g in parts:
        for e in g.entities:
            key = e.name.casefold()
            node = nodes.setdefault(key, {"id": key, "name": e.name, "type": e.type, "mentions": 0})
            node["mentions"] += 1
        for r in g.relations:
            src, dst = r.source.casefold(), r.target.casefold()
            for key, name in ((src, r.source), (dst, r.target)):
                nodes.setdefault(key, {"id": key, "name": name, "type": "concept", "mentions": 0})
            edge = edges.setdefault(
                (src, dst, r.predicate),
                {
                    "source": src,
                    "target": dst,
                    "predicate": r.predicate,
                    "sections": [],
                    "pages": pages.get(r.section),
                },
            )
            edge["sections"].append(r.section)
    graph = {"nodes": list(nodes.values()), "edges": list(edges.values())}
    doc.graph = graph
    await session.commit()
    return graph

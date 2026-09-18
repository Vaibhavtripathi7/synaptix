"""Map-reduce document summary over sections. Sections are packed into batches that fit the
map prompt; each batch is summarised concurrently, then the partials are reduced into one
summary. Faithfulness beats brevity: numbers and names are kept."""

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, Section
from app.ingest.chunk import count_tokens
from app.llm.client import Gemini

_BATCH_TOKENS = 12_000
_MAP = (
    "Summarise the following part of a document in 5-8 sentences. Keep every concrete "
    "figure, name and conclusion; drop boilerplate. Plain prose.\n\n"
)
_REDUCE = (
    "Combine these partial summaries of one document into a single coherent summary of about "
    "250 words: what the document is, its key content, and its main findings or conclusions. "
    "Use Markdown with a short lead paragraph followed by bullet points.\n\n"
)


async def summarize_document(session: AsyncSession, doc: Document, llm: Gemini) -> str:
    sections = list(
        await session.scalars(
            select(Section).where(Section.document_id == doc.id).order_by(Section.order_idx)
        )
    )
    batches: list[list[str]] = [[]]
    spent = 0
    for s in sections:
        text = f"## {' > '.join(s.heading_path)}\n{s.content}"
        cost = count_tokens(text)
        if spent + cost > _BATCH_TOKENS and batches[-1]:
            batches.append([])
            spent = 0
        batches[-1].append(text)
        spent += cost

    partials = await asyncio.gather(
        *(llm.generate(_MAP + "\n\n".join(b), think=True) for b in batches if b)
    )
    summary = (
        partials[0]
        if len(partials) == 1
        else await llm.generate(_REDUCE + "\n\n---\n\n".join(partials), think=True)
    )
    doc.summary = summary
    await session.commit()
    return summary

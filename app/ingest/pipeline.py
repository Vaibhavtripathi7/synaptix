"""Ingestion job: parse → enrich → chunk → embed → persist. Status is committed after each stage
so the API can report progress; failure records the error and leaves the source PDF in place."""

import asyncio
import uuid

import structlog
from sqlalchemy import delete

from app.core.settings import Settings
from app.db.models import Chunk, DocStatus, Document, Element, ElementKind, Section
from app.db.session import SessionLocal
from app.ingest.chunk import build_sections, chunk_sections
from app.ingest.enrich import enrich
from app.ingest.parse import parse_pdf
from app.llm.client import Gemini
from app.llm.embeddings import Embedder
from app.storage import LocalStorage

log = structlog.get_logger()


async def ingest_document(
    document_id: uuid.UUID,
    *,
    storage: LocalStorage,
    llm: Gemini,
    embedder: Embedder,
    settings: Settings,
) -> None:
    async with SessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            log.warning("ingest.missing", document_id=str(document_id))
            return
        try:
            await _run(session, doc, storage, llm, embedder, settings)
        except asyncio.CancelledError:
            # Worker shutdown or restart. Record it before unwinding, shielded so the write
            # itself is not cancelled; a SIGKILL skips this and the startup reaper covers it.
            log.warning("ingest.cancelled", document_id=str(document_id))
            doc.status, doc.error = DocStatus.failed, "ingestion was interrupted; upload again"
            await asyncio.shield(session.commit())
            raise
        except Exception as exc:
            log.exception("ingest.failed", document_id=str(document_id))
            doc.status, doc.error = DocStatus.failed, f"{type(exc).__name__}: {exc}"[:2000]
            await session.commit()
            raise


async def _run(session, doc: Document, storage, llm, embedder, settings: Settings) -> None:
    async def stage(status: DocStatus) -> None:
        doc.status, doc.error = status, None
        await session.commit()
        log.info("ingest.stage", document_id=str(doc.id), stage=status.value)

    await stage(DocStatus.parsing)
    parsed = await asyncio.to_thread(parse_pdf, await storage.get(doc.storage_key))
    doc.title, doc.page_count = parsed.title or doc.filename, parsed.page_count

    await stage(DocStatus.enriching)
    degraded = await enrich(parsed, llm, settings.ocr_batch_pages)

    sections = build_sections(parsed.blocks, parsed.title)
    drafts = chunk_sections(
        parsed.blocks, sections, settings.chunk_tokens, settings.chunk_overlap_tokens
    )

    await stage(DocStatus.embedding)
    vectors = await embedder.embed_documents([c.content for c in drafts])

    # Replace any previous extraction (re-ingest) in one transaction.
    for model in (Chunk, Section, Element):
        await session.execute(delete(model).where(model.document_id == doc.id))

    elements: list[Element] = []
    for i, b in enumerate(parsed.blocks):
        meta = dict(b.meta, role=b.role)
        if b.image_png is not None:
            meta["image_key"] = f"{doc.id}/figures/p{b.page}_{i}.png"
            await storage.put(meta["image_key"], b.image_png)
        elements.append(
            Element(
                document_id=doc.id,
                page=b.page,
                order_idx=i,
                kind=b.kind,
                content=b.content,
                bbox=list(b.bbox) if b.bbox else None,
                meta=meta,
            )
        )
    section_rows = [
        Section(
            document_id=doc.id,
            order_idx=i,
            heading_path=s.heading_path,
            page_start=s.page_start,
            page_end=s.page_end,
            content="\n\n".join(parsed.blocks[j].content for j in s.block_idx),
        )
        for i, s in enumerate(sections)
    ]
    session.add_all(elements + section_rows)
    await session.flush()  # assigns ids for the FK/array references below

    session.add_all(
        Chunk(
            document_id=doc.id,
            section_id=section_rows[c.section_idx].id,
            order_idx=i,
            kind=c.kind,
            content=c.content,
            token_count=c.token_count,
            page_start=c.page_start,
            page_end=c.page_end,
            element_ids=[elements[j].id for j in c.block_idx],
            embedding=vec,
        )
        for i, (c, vec) in enumerate(zip(drafts, vectors, strict=True))
    )
    doc.meta = {
        **doc.meta,
        "elements": len(elements),
        "sections": len(section_rows),
        "chunks": len(drafts),
        "figures": sum(b.kind is ElementKind.figure for b in parsed.blocks),
        "tables": sum(b.kind is ElementKind.table for b in parsed.blocks),
        "enrich_degraded": degraded,
    }
    await stage(DocStatus.ready)

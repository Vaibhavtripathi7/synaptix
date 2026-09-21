import hashlib
import uuid

import pymupdf
from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import QueueDep, SessionDep, SettingsDep, StorageDep
from app.api.schemas import DocumentOut, ElementOut
from app.db.models import Chunk, DocStatus, Document, Element

router = APIRouter(prefix="/documents", tags=["documents"])


async def get_or_404(session: AsyncSession, document_id: uuid.UUID) -> Document:
    doc = await session.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    return doc


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentOut)
async def upload(
    file: UploadFile,
    session: SessionDep,
    storage: StorageDep,
    queue: QueueDep,
    settings: SettingsDep,
) -> Document:
    """Accepts a PDF, deduplicates it by content hash, and queues ingestion.
    Re-uploading a known document returns it unchanged unless its last ingest failed."""
    if file.content_type != "application/pdf":
        raise HTTPException(415, "only application/pdf is accepted")
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"file exceeds {settings.max_upload_mb} MB")
    try:
        pymupdf.open(stream=data, filetype="pdf").close()
    except Exception:
        raise HTTPException(415, "file is not a readable PDF") from None

    digest = hashlib.sha256(data).hexdigest()
    doc = await session.scalar(select(Document).where(Document.content_hash == digest))
    if doc is not None and doc.status is not DocStatus.failed:
        return doc
    if doc is None:
        doc = Document(
            id=uuid.uuid4(),
            filename=(file.filename or "document.pdf")[:255],
            content_hash=digest,
            storage_key="",
        )
        doc.storage_key = f"{doc.id}/source.pdf"
        session.add(doc)
    doc.status, doc.error = DocStatus.queued, None
    await session.commit()
    await storage.put(doc.storage_key, data)
    await queue.enqueue_job("ingest", str(doc.id))
    return doc


@router.get("", response_model=list[DocumentOut])
async def list_documents(session: SessionDep) -> list[Document]:
    rows = await session.scalars(select(Document).order_by(Document.created_at.desc()))
    return list(rows)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: uuid.UUID, session: SessionDep) -> Document:
    return await get_or_404(session, document_id)


@router.get("/{document_id}/file")
async def get_file(
    document_id: uuid.UUID, session: SessionDep, storage: StorageDep
) -> FileResponse:
    doc = await get_or_404(session, document_id)
    # FileResponse handles Range requests, so the viewer fetches pages as it needs them.
    return FileResponse(
        storage.path(doc.storage_key),
        media_type="application/pdf",
        filename=doc.filename,
        content_disposition_type="inline",
    )


@router.get("/{document_id}/figures/{name}")
async def get_figure(
    document_id: uuid.UUID,
    name: str,
    session: SessionDep,
    storage: StorageDep,
) -> FileResponse:
    await get_or_404(session, document_id)
    path = storage.path(f"{document_id}/figures/{name}")
    if not path.is_file():
        raise HTTPException(404, "figure not found")
    return FileResponse(path, media_type="image/png")


@router.get("/{document_id}/chunks/{chunk_id}", response_model=list[ElementOut])
async def chunk_elements(
    document_id: uuid.UUID, chunk_id: uuid.UUID, session: SessionDep
) -> list[Element]:
    """The page elements (with bboxes) a chunk was built from: what a citation highlights."""
    await get_or_404(session, document_id)
    chunk = await session.get(Chunk, chunk_id)
    if chunk is None or chunk.document_id != document_id:
        raise HTTPException(404, "chunk not found")
    rows = await session.scalars(
        select(Element).where(Element.id.in_(chunk.element_ids)).order_by(Element.order_idx)
    )
    return list(rows)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    session: SessionDep,
    storage: StorageDep,
) -> None:
    doc = await get_or_404(session, document_id)
    await session.delete(doc)
    await session.commit()
    await storage.delete_prefix(f"{document_id}/")

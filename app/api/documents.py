import hashlib
import uuid

import pymupdf
from fastapi import APIRouter, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    AnswererDep,
    GeminiKeyDep,
    QueueDep,
    ReadLimit,
    SessionDep,
    SettingsDep,
    StorageDep,
    UploadLimit,
    WorkspaceDep,
)
from app.api.schemas import DocumentOut, ElementOut
from app.core.middleware import current_request_id
from app.db.models import Chunk, DocStatus, Document, Element
from app.db.session import end_read

router = APIRouter(prefix="/documents", tags=["documents"])


async def owned_document(session: AsyncSession, workspace: str, document_id: uuid.UUID) -> Document:
    """The one ownership check every document route goes through. A document in another
    workspace is reported as missing, not forbidden: existence is not disclosed."""
    doc = await session.scalar(
        select(Document).where(Document.id == document_id, Document.workspace_id == workspace)
    )
    if doc is None:
        raise HTTPException(404, "document not found")
    return doc


@router.post(
    "", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentOut, dependencies=[UploadLimit]
)
async def upload(
    file: UploadFile,
    workspace: WorkspaceDep,
    key: GeminiKeyDep,
    _: AnswererDep,  # fails fast with 400 if no Gemini key is available for the ingest
    session: SessionDep,
    storage: StorageDep,
    queue: QueueDep,
    settings: SettingsDep,
) -> Document:
    """Accepts a PDF, deduplicates it within the workspace, and queues ingestion.
    Re-uploading a known document returns it unchanged unless its last ingest failed."""
    if file.content_type != "application/pdf":
        raise HTTPException(415, "only application/pdf is accepted")
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"file exceeds {settings.max_upload_mb} MB")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            pages = pdf.page_count
    except Exception:
        raise HTTPException(415, "file is not a readable PDF") from None
    if pages > settings.max_pages:
        raise HTTPException(413, f"{pages} pages; the limit is {settings.max_pages}")

    digest = hashlib.sha256(data).hexdigest()
    doc = await session.scalar(
        select(Document).where(Document.workspace_id == workspace, Document.content_hash == digest)
    )
    if doc is not None and doc.status is not DocStatus.failed:
        return doc
    if doc is None:
        count = await session.scalar(
            select(func.count()).select_from(Document).where(Document.workspace_id == workspace)
        )
        if count >= settings.max_documents_per_workspace:
            raise HTTPException(409, f"workspace holds {count} documents; delete one to add more")
        doc = Document(
            id=uuid.uuid4(),
            workspace_id=workspace,
            filename=(file.filename or "document.pdf")[:255],
            content_hash=digest,
            storage_key="",
        )
        doc.storage_key = f"{doc.id}/source.pdf"
        session.add(doc)
    doc.status, doc.error = DocStatus.queued, None
    try:
        await session.commit()
    except IntegrityError:
        # A concurrent upload of the same file into the same workspace won the insert: the
        # unique (workspace, hash) constraint is the arbiter, so return the winner's row.
        await session.rollback()
        return await session.scalar(
            select(Document).where(
                Document.workspace_id == workspace, Document.content_hash == digest
            )
        )
    await storage.put(doc.storage_key, data)  # after the commit: a lost race writes nothing
    # The key rides the job payload and is never written to Postgres. Redis runs without
    # persistence and the worker keeps no job results, so it is gone once the job finishes.
    await queue.enqueue_job(
        "ingest",
        str(doc.id),
        key,
        request_id=current_request_id(),
        _job_id=f"ingest:{doc.id}",  # arq refuses a duplicate while one is queued or running
    )
    return doc


@router.get("", response_model=list[DocumentOut], dependencies=[ReadLimit])
async def list_documents(workspace: WorkspaceDep, session: SessionDep) -> list[Document]:
    rows = await session.scalars(
        select(Document)
        .where(Document.workspace_id == workspace)
        .order_by(Document.created_at.desc())
    )
    return list(rows)


@router.get("/{document_id}", response_model=DocumentOut, dependencies=[ReadLimit])
async def get_document(
    document_id: uuid.UUID, workspace: WorkspaceDep, session: SessionDep
) -> Document:
    return await owned_document(session, workspace, document_id)


@router.get("/{document_id}/file", dependencies=[ReadLimit])
async def get_file(
    document_id: uuid.UUID, workspace: WorkspaceDep, session: SessionDep, storage: StorageDep
) -> FileResponse:
    doc = await owned_document(session, workspace, document_id)
    await end_read(session)  # streaming a large file must not hold a transaction
    # FileResponse handles Range requests, so the viewer fetches pages as it needs them.
    return FileResponse(
        storage.path(doc.storage_key),
        media_type="application/pdf",
        filename=doc.filename,
        content_disposition_type="inline",
    )


@router.get("/{document_id}/figures/{name}", dependencies=[ReadLimit])
async def get_figure(
    document_id: uuid.UUID,
    name: str,
    workspace: WorkspaceDep,
    session: SessionDep,
    storage: StorageDep,
) -> FileResponse:
    await owned_document(session, workspace, document_id)
    await end_read(session)
    path = storage.path(f"{document_id}/figures/{name}")
    if not path.is_file():
        raise HTTPException(404, "figure not found")
    return FileResponse(path, media_type="image/png")


@router.get(
    "/{document_id}/chunks/{chunk_id}", response_model=list[ElementOut], dependencies=[ReadLimit]
)
async def chunk_elements(
    document_id: uuid.UUID, chunk_id: uuid.UUID, workspace: WorkspaceDep, session: SessionDep
) -> list[Element]:
    """The page elements (with bboxes) a chunk was built from: what a citation highlights."""
    await owned_document(session, workspace, document_id)
    chunk = await session.get(Chunk, chunk_id)
    if chunk is None or chunk.document_id != document_id:
        raise HTTPException(404, "chunk not found")
    rows = await session.scalars(
        select(Element).where(Element.id.in_(chunk.element_ids)).order_by(Element.order_idx)
    )
    return list(rows)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[ReadLimit])
async def delete_document(
    document_id: uuid.UUID,
    workspace: WorkspaceDep,
    session: SessionDep,
    storage: StorageDep,
) -> None:
    doc = await owned_document(session, workspace, document_id)
    await session.delete(doc)
    await session.commit()
    await storage.delete_prefix(f"{document_id}/")

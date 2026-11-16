import hashlib
import re
import uuid

import pymupdf
from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    AnswerCacheDep,
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
from app.core.settings import Settings
from app.core.validation import clean_filename
from app.db.models import Chunk, DocStatus, Document, Element
from app.db.session import end_read

router = APIRouter(prefix="/documents", tags=["documents"])
_FIGURE_NAME = re.compile(r"p\d+_\d+\.png")


async def owned_document(session: AsyncSession, workspace: str, document_id: uuid.UUID) -> Document:
    """The one ownership check every document route goes through. A document in another
    workspace is reported as missing, not forbidden: existence is not disclosed."""
    doc = await session.scalar(
        select(Document).where(Document.id == document_id, Document.workspace_id == workspace)
    )
    if doc is None:
        raise HTTPException(404, "document not found")
    return doc


async def _enforce_quotas(
    session: AsyncSession, workspace: str, incoming: int, settings: Settings
) -> None:
    """Document count and bytes per workspace, and bytes across everything. Soft limits: two
    concurrent uploads can overshoot by one file, which bounds the error at one upload's size."""
    count, used = (
        await session.execute(
            select(func.count(), func.coalesce(func.sum(Document.size_bytes), 0)).where(
                Document.workspace_id == workspace
            )
        )
    ).one()
    if count >= settings.max_documents_per_workspace:
        raise HTTPException(409, f"workspace holds {count} documents; delete one to add more")
    if used + incoming > settings.max_workspace_bytes:
        raise HTTPException(413, "workspace storage quota reached; delete a document to add more")
    total = await session.scalar(select(func.coalesce(func.sum(Document.size_bytes), 0)))
    if total + incoming > settings.max_total_bytes:
        raise HTTPException(507, "the service is out of storage for new documents; try later")


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
        await _enforce_quotas(session, workspace, len(data), settings)
        doc = Document(
            id=uuid.uuid4(),
            size_bytes=len(data),
            workspace_id=workspace,
            filename=clean_filename(file.filename),
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
    await storage.put(
        doc.storage_key, data, "application/pdf"
    )  # after the commit: a lost race writes nothing
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
async def list_documents(workspace: WorkspaceDep, session: SessionDep) -> list:
    """Hot path - the UI polls it. Selects exactly the columns returned, as plain rows: building
    ORM entities was the dominant per-request cost (measured, docs/capacity.md)."""
    columns = [getattr(Document, name) for name in DocumentOut.model_fields]
    result = await session.execute(
        select(*columns)
        .where(Document.workspace_id == workspace)
        .order_by(Document.created_at.desc())
    )
    return list(result.all())


@router.get("/{document_id}", response_model=DocumentOut, dependencies=[ReadLimit])
async def get_document(
    document_id: uuid.UUID, workspace: WorkspaceDep, session: SessionDep
) -> Document:
    return await owned_document(session, workspace, document_id)


@router.get("/{document_id}/file", dependencies=[ReadLimit])
async def get_file(
    document_id: uuid.UUID,
    request: Request,
    workspace: WorkspaceDep,
    session: SessionDep,
    storage: StorageDep,
) -> Response:
    doc = await owned_document(session, workspace, document_id)
    await end_read(session)  # streaming a large file must not hold a transaction
    return await storage.response(
        doc.storage_key, "application/pdf", request.headers.get("range"), filename=doc.filename
    )


@router.get("/{document_id}/figures/{name}", dependencies=[ReadLimit])
async def get_figure(
    document_id: uuid.UUID,
    name: str,
    request: Request,
    workspace: WorkspaceDep,
    session: SessionDep,
    storage: StorageDep,
) -> Response:
    # Figure names are always p<page>_<index>.png. Anything else - dot segments, encoded
    # separators - could address another document's object once keys reach a bucket.
    if not _FIGURE_NAME.fullmatch(name):
        raise HTTPException(404, "figure not found")
    await owned_document(session, workspace, document_id)
    await end_read(session)
    return await storage.response(
        f"{document_id}/figures/{name}", "image/png", request.headers.get("range")
    )


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
    cache: AnswerCacheDep,
) -> None:
    doc = await owned_document(session, workspace, document_id)
    await session.delete(doc)
    await session.commit()
    await cache.invalidate(workspace)  # after the commit: answers citing it must not be served
    await storage.delete_prefix(f"{document_id}/")

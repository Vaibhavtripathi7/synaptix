import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from app.api.deps import (
    AnswererDep,
    GeminiKeyDep,
    QueueDep,
    SessionDep,
    WorkspaceDep,
)
from app.db.models import AnalysisKind, AnalysisRun, DocStatus, Document, RunStatus

router = APIRouter(prefix="/analyses", tags=["analyses"])


class AnalysisRequest(BaseModel):
    kind: AnalysisKind
    document_ids: list[uuid.UUID] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def _arity(self) -> "AnalysisRequest":
        need = 2 if self.kind is AnalysisKind.comparison else 1
        if len(self.document_ids) != need:
            raise ValueError(f"{self.kind.value} analysis takes exactly {need} document id(s)")
        return self


class AnalysisOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: AnalysisKind
    document_ids: list[uuid.UUID]
    status: RunStatus
    result: dict | None
    usage: dict | None
    error: str | None
    created_at: datetime
    finished_at: datetime | None


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=AnalysisOut)
async def create(
    body: AnalysisRequest,
    workspace: WorkspaceDep,
    key: GeminiKeyDep,
    _: AnswererDep,  # 400 now, rather than a failed job later, if no key is available
    session: SessionDep,
    queue: QueueDep,
) -> AnalysisRun:
    """Queues an analysis DAG over ready documents. Poll GET /analyses/{id} for the result."""
    docs = list(
        await session.scalars(
            select(Document).where(
                Document.id.in_(body.document_ids), Document.workspace_id == workspace
            )
        )
    )
    missing = set(body.document_ids) - {d.id for d in docs}
    if missing:
        raise HTTPException(404, f"unknown document(s): {', '.join(map(str, missing))}")
    not_ready = [str(d.id) for d in docs if d.status is not DocStatus.ready]
    if not_ready:
        raise HTTPException(409, f"document(s) not ready: {', '.join(not_ready)}")
    run = AnalysisRun(workspace_id=workspace, kind=body.kind, document_ids=body.document_ids)
    session.add(run)
    await session.commit()
    await queue.enqueue_job("analyze", str(run.id), key)
    return run


@router.get("", response_model=list[AnalysisOut])
async def list_runs(
    workspace: WorkspaceDep, session: SessionDep, document_id: uuid.UUID | None = None
) -> list[AnalysisRun]:
    stmt = (
        select(AnalysisRun)
        .where(AnalysisRun.workspace_id == workspace)
        .order_by(AnalysisRun.created_at.desc())
    )
    if document_id:
        stmt = stmt.where(AnalysisRun.document_ids.any(document_id))
    return list(await session.scalars(stmt))


@router.get("/{run_id}", response_model=AnalysisOut)
async def get_run(run_id: uuid.UUID, workspace: WorkspaceDep, session: SessionDep) -> AnalysisRun:
    run = await session.get(AnalysisRun, run_id)
    if run is None or run.workspace_id != workspace:
        raise HTTPException(404, "analysis not found")
    return run

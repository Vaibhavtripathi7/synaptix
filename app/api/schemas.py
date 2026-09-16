import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.db.models import DocStatus, ElementKind


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    title: str | None
    status: DocStatus
    error: str | None
    page_count: int | None
    summary: str | None
    meta: dict
    created_at: datetime
    updated_at: datetime


class ElementOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    page: int
    kind: ElementKind
    bbox: list[float] | None
    content: str
    meta: dict

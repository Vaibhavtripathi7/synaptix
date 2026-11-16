"""Relational schema. One Postgres holds documents, structure, chunks, vectors and FTS."""

import uuid
from datetime import datetime
from enum import StrEnum

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Computed,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.settings import get_settings


class Base(DeclarativeBase):
    pass


class DocStatus(StrEnum):
    queued = "queued"
    parsing = "parsing"
    enriching = "enriching"
    embedding = "embedding"
    ready = "ready"
    failed = "failed"


class ElementKind(StrEnum):
    text = "text"
    table = "table"
    figure = "figure"


class AnalysisKind(StrEnum):
    study_guide = "study_guide"  # outline -> practice questions, over one document
    comparison = "comparison"  # two documents compared


class RunStatus(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Document(Base):
    __tablename__ = "documents"
    # Fetch server-side timestamps in the same flush (RETURNING) so a committed row can be
    # serialised without a lazy refresh, which is illegal outside the async session context.
    __mapper_args__ = {"eager_defaults": True}
    # Dedupe is per workspace: a global unique hash would hand one tenant's upload to another.
    __table_args__ = (UniqueConstraint("workspace_id", "content_hash"),)

    id: Mapped[uuid.UUID] = _uuid_pk()
    # sha256 of the workspace token: the owner. Every document route and search filters on it.
    workspace_id: Mapped[str] = mapped_column(String(64), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    content_hash: Mapped[str] = mapped_column(String(64))
    storage_key: Mapped[str] = mapped_column(String(255))
    page_count: Mapped[int | None] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)  # source PDF, for quotas
    status: Mapped[DocStatus] = mapped_column(
        Enum(DocStatus, name="doc_status"), default=DocStatus.queued
    )
    error: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(String(512))
    summary: Mapped[str | None] = mapped_column(Text)
    # Deferred: an entity graph can run to hundreds of kilobytes, and listing a library must
    # not drag every document's graph out of Postgres just to render titles.
    graph: Mapped[dict | None] = mapped_column(JSONB, deferred=True)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    # No `chunks` collection: nothing reads it, and an ORM cascade would load every chunk (with
    # its embedding) before deleting a document. The ON DELETE CASCADE foreign keys remove a
    # document's elements, sections and chunks in the database instead.


class Element(Base):
    """A typed unit extracted from a page: a text block, a table, or a figure."""

    __tablename__ = "elements"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page: Mapped[int] = mapped_column(Integer)
    order_idx: Mapped[int] = mapped_column(Integer)
    kind: Mapped[ElementKind] = mapped_column(Enum(ElementKind, name="element_kind"))
    content: Mapped[str] = mapped_column(Text)  # markdown text / table markdown / figure caption
    bbox: Mapped[list[float] | None] = mapped_column(ARRAY(Float))  # x0,y0,x1,y1 in PDF points
    meta: Mapped[dict] = mapped_column(JSONB, default=dict)  # table rows, image storage key, ...


class Section(Base):
    """Heading-delimited span of a document; the 'parent' in small-to-big retrieval."""

    __tablename__ = "sections"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    order_idx: Mapped[int] = mapped_column(Integer)
    heading_path: Mapped[list[str]] = mapped_column(ARRAY(Text))
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)


class Chunk(Base):
    """Retrieval unit. Dense vector + generated tsvector give hybrid search from one table."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = _uuid_pk()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    section_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("sections.id", ondelete="CASCADE")
    )
    order_idx: Mapped[int] = mapped_column(Integer)
    kind: Mapped[ElementKind] = mapped_column(
        Enum(ElementKind, name="element_kind", create_type=False)
    )
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    element_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)), default=list)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(get_settings().embedding_dim))
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )

    document: Mapped[Document] = relationship()
    section: Mapped[Section | None] = relationship()

    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class AnalysisRun(Base):
    """One execution of an analysis DAG over one or two documents; result is stage -> JSON."""

    __tablename__ = "analysis_runs"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[uuid.UUID] = _uuid_pk()
    workspace_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[AnalysisKind] = mapped_column(Enum(AnalysisKind, name="analysis_kind"))
    document_ids: Mapped[list[uuid.UUID]] = mapped_column(ARRAY(UUID(as_uuid=True)))
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus, name="run_status"), default=RunStatus.queued
    )
    result: Mapped[dict | None] = mapped_column(JSONB)
    usage: Mapped[dict | None] = mapped_column(JSONB)  # tokens the run consumed
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

"""initial schema: documents, elements, sections, chunks (pgvector + FTS)

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql as pg

from app.core.settings import get_settings

revision = "0001"
down_revision = None

DIM = get_settings().embedding_dim


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    doc_status = pg.ENUM(
        "queued", "parsing", "enriching", "embedding", "ready", "failed", name="doc_status"
    )
    element_kind = pg.ENUM("text", "table", "figure", name="element_kind")

    op.create_table(
        "documents",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("storage_key", sa.String(255), nullable=False),
        sa.Column("page_count", sa.Integer),
        sa.Column("status", doc_status, nullable=False),
        sa.Column("error", sa.Text),
        sa.Column("title", sa.String(512)),
        sa.Column("summary", sa.Text),
        sa.Column("meta", pg.JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_table(
        "elements",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("page", sa.Integer, nullable=False),
        sa.Column("order_idx", sa.Integer, nullable=False),
        sa.Column("kind", element_kind, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("bbox", pg.ARRAY(sa.Float)),
        sa.Column("meta", pg.JSONB, nullable=False, server_default="{}"),
    )
    op.create_table(
        "sections",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("order_idx", sa.Integer, nullable=False),
        sa.Column("heading_path", pg.ARRAY(sa.Text), nullable=False),
        sa.Column("page_start", sa.Integer, nullable=False),
        sa.Column("page_end", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
    )
    op.create_table(
        "chunks",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            pg.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "section_id", pg.UUID(as_uuid=True), sa.ForeignKey("sections.id", ondelete="CASCADE")
        ),
        sa.Column("order_idx", sa.Integer, nullable=False),
        sa.Column("kind", element_kind, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False),
        sa.Column("page_start", sa.Integer, nullable=False),
        sa.Column("page_end", sa.Integer, nullable=False),
        sa.Column(
            "element_ids", pg.ARRAY(pg.UUID(as_uuid=True)), nullable=False, server_default="{}"
        ),
        sa.Column("embedding", Vector(DIM)),
        sa.Column(
            "tsv", pg.TSVECTOR, sa.Computed("to_tsvector('english', content)", persisted=True)
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    for table in ("chunks", "sections", "elements", "documents"):
        op.drop_table(table)
    op.execute("DROP TYPE element_kind; DROP TYPE doc_status")

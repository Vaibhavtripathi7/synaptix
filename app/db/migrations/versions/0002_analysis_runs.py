"""analysis runs

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0002"
down_revision = "0001"


def upgrade() -> None:
    op.create_table(
        "analysis_runs",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "kind", pg.ENUM("competitor", "comparison", name="analysis_kind"), nullable=False
        ),
        sa.Column("document_ids", pg.ARRAY(pg.UUID(as_uuid=True)), nullable=False),
        sa.Column(
            "status",
            pg.ENUM("queued", "running", "done", "failed", name="run_status"),
            nullable=False,
        ),
        sa.Column("result", pg.JSONB),
        sa.Column("error", sa.Text),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_analysis_runs_document_ids", "analysis_runs", ["document_ids"], postgresql_using="gin"
    )


def downgrade() -> None:
    op.drop_table("analysis_runs")
    for enum in ("analysis_kind", "run_status"):
        op.execute(f"DROP TYPE {enum}")

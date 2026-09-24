"""workspaces: every document and analysis run belongs to one

Revision ID: 0004
Revises: 0003

Existing rows get a random workspace each, so they become unreachable rather than readable by
anyone: backfilling a shared or guessable value would publish them.
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"

_RANDOM_OWNER = "encode(sha256(gen_random_uuid()::text::bytea), 'hex')"


def upgrade() -> None:
    for table in ("documents", "analysis_runs"):
        op.add_column(table, sa.Column("workspace_id", sa.String(64)))
        op.execute(f"UPDATE {table} SET workspace_id = {_RANDOM_OWNER}")
        op.alter_column(table, "workspace_id", nullable=False)
        op.create_index(f"ix_{table}_workspace_id", table, ["workspace_id"])
    op.drop_constraint("documents_content_hash_key", "documents", type_="unique")
    op.create_unique_constraint(
        "uq_documents_workspace_id", "documents", ["workspace_id", "content_hash"]
    )


def downgrade() -> None:
    op.drop_constraint("uq_documents_workspace_id", "documents", type_="unique")
    op.create_unique_constraint("documents_content_hash_key", "documents", ["content_hash"])
    for table in ("documents", "analysis_runs"):
        op.drop_index(f"ix_{table}_workspace_id", table)
        op.drop_column(table, "workspace_id")

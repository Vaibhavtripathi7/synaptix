"""entity graph cached on the document

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0003"
down_revision = "0002"


def upgrade() -> None:
    op.add_column("documents", sa.Column("graph", pg.JSONB))


def downgrade() -> None:
    op.drop_column("documents", "graph")

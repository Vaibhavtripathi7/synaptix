"""token usage recorded on analysis runs

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0005"
down_revision = "0004"


def upgrade() -> None:
    op.add_column("analysis_runs", sa.Column("usage", pg.JSONB))


def downgrade() -> None:
    op.drop_column("analysis_runs", "usage")

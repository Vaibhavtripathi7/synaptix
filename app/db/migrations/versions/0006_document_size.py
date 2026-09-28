"""document size, for storage quotas

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"


def upgrade() -> None:
    op.add_column(
        "documents", sa.Column("size_bytes", sa.BigInteger, nullable=False, server_default="0")
    )


def downgrade() -> None:
    op.drop_column("documents", "size_bytes")

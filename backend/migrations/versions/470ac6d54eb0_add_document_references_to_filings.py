"""add document references to filings

Revision ID: 470ac6d54eb0
Revises: cf83b415ae15
Create Date: 2026-08-20 13:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "470ac6d54eb0"
down_revision: str | Sequence[str] | None = "cf83b415ae15"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "filings",
        sa.Column("source_document_id", sa.Text(), nullable=True),
    )
    op.add_column(
        "filings",
        sa.Column("document_metadata_url", sa.Text(), nullable=True),
    )
    op.create_index(
        op.f("ix_filings_source_document_id"),
        "filings",
        ["source_document_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_filings_source_document_id"), table_name="filings")
    op.drop_column("filings", "document_metadata_url")
    op.drop_column("filings", "source_document_id")

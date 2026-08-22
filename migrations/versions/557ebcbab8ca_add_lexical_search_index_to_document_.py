"""add lexical search index to document pages

Revision ID: 557ebcbab8ca
Revises: b9f1038e67a2
Create Date: 2026-08-22 16:25:53.198182

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "557ebcbab8ca"
down_revision: str | Sequence[str] | None = "b9f1038e67a2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(
        "CREATE INDEX ix_document_pages_search_vector "
        "ON document_pages USING gin (to_tsvector('english', text))"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX ix_document_pages_search_vector")

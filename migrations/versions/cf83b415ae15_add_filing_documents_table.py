"""add filing documents table

Revision ID: cf83b415ae15
Revises: 5f66dd79992f
Create Date: 2026-08-20 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "cf83b415ae15"
down_revision: str | Sequence[str] | None = "5f66dd79992f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "filing_documents",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("filing_id", sa.BigInteger(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("source_document_id", sa.Text(), nullable=False),
        sa.Column("media_type", sa.Text(), nullable=False),
        sa.Column("content_length", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("etag", sa.Text(), nullable=True),
        sa.Column("pages", sa.Integer(), nullable=True),
        sa.Column("source_created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "raw_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("first_retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "content_length > 0",
            name="ck_filing_documents_content_length_positive",
        ),
        sa.CheckConstraint(
            "pages IS NULL OR pages >= 0",
            name="ck_filing_documents_pages_non_negative",
        ),
        sa.CheckConstraint(
            "char_length(sha256) = 64",
            name="ck_filing_documents_sha256_length",
        ),
        sa.ForeignKeyConstraint(
            ["filing_id"],
            ["filings.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source",
            "source_document_id",
            "sha256",
            name="uq_filing_documents_source_document_sha256",
        ),
    )
    op.create_index(
        op.f("ix_filing_documents_filing_id"),
        "filing_documents",
        ["filing_id"],
        unique=False,
    )
    op.create_index(
        "ix_filing_documents_source_document_id",
        "filing_documents",
        ["source", "source_document_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_filing_documents_source_document_id",
        table_name="filing_documents",
    )
    op.drop_index(
        op.f("ix_filing_documents_filing_id"),
        table_name="filing_documents",
    )
    op.drop_table("filing_documents")

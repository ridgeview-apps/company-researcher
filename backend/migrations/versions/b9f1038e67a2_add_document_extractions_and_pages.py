"""add document extractions and pages

Revision ID: b9f1038e67a2
Revises: 470ac6d54eb0
Create Date: 2026-08-20 17:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b9f1038e67a2"
down_revision: str | Sequence[str] | None = "470ac6d54eb0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "document_extractions",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("filing_document_id", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("extractor", sa.Text(), nullable=False),
        sa.Column("extractor_version", sa.Text(), nullable=False),
        sa.Column("renderer", sa.Text(), nullable=False),
        sa.Column("renderer_version", sa.Text(), nullable=False),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column("render_dpi", sa.Integer(), nullable=False),
        sa.Column("page_segmentation_mode", sa.Integer(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("total_character_count", sa.BigInteger(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "total_character_count IS NULL OR total_character_count >= 0",
            name="ck_document_extractions_character_count_non_negative",
        ),
        sa.CheckConstraint(
            "page_count IS NULL OR page_count >= 0",
            name="ck_document_extractions_page_count_non_negative",
        ),
        sa.CheckConstraint(
            "page_segmentation_mode BETWEEN 0 AND 13",
            name="ck_document_extractions_page_segmentation_mode",
        ),
        sa.CheckConstraint(
            "render_dpi >= 72",
            name="ck_document_extractions_render_dpi_minimum",
        ),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed')",
            name="ck_document_extractions_status",
        ),
        sa.ForeignKeyConstraint(
            ["filing_document_id"],
            ["filing_documents.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "filing_document_id",
            "extractor",
            "extractor_version",
            "renderer",
            "renderer_version",
            "language",
            "render_dpi",
            "page_segmentation_mode",
            name="uq_document_extractions_document_configuration",
        ),
    )
    op.create_index(
        op.f("ix_document_extractions_filing_document_id"),
        "document_extractions",
        ["filing_document_id"],
        unique=False,
    )
    op.create_table(
        "document_pages",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("document_extraction_id", sa.BigInteger(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("character_count", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "character_count >= 0",
            name="ck_document_pages_character_count_non_negative",
        ),
        sa.CheckConstraint(
            "page_number >= 1",
            name="ck_document_pages_page_number_positive",
        ),
        sa.ForeignKeyConstraint(
            ["document_extraction_id"],
            ["document_extractions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_extraction_id",
            "page_number",
            name="uq_document_pages_extraction_page_number",
        ),
    )
    op.create_index(
        op.f("ix_document_pages_document_extraction_id"),
        "document_pages",
        ["document_extraction_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_document_pages_document_extraction_id"),
        table_name="document_pages",
    )
    op.drop_table("document_pages")
    op.drop_index(
        op.f("ix_document_extractions_filing_document_id"),
        table_name="document_extractions",
    )
    op.drop_table("document_extractions")

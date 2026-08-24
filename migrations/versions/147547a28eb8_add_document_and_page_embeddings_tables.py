"""add document and page embeddings tables

Revision ID: 147547a28eb8
Revises: 557ebcbab8ca
Create Date: 2026-08-24 14:28:57.764545

"""

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "147547a28eb8"
down_revision: str | Sequence[str] | None = "557ebcbab8ca"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSIONS = 1536


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "document_embeddings",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("document_extraction_id", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=True),
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
            "dimensions > 0",
            name="ck_document_embeddings_dimensions_positive",
        ),
        sa.CheckConstraint(
            "page_count IS NULL OR page_count >= 0",
            name="ck_document_embeddings_page_count_non_negative",
        ),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed')",
            name="ck_document_embeddings_status",
        ),
        sa.ForeignKeyConstraint(
            ["document_extraction_id"],
            ["document_extractions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_extraction_id",
            "provider",
            "model",
            "dimensions",
            name="uq_document_embeddings_extraction_configuration",
        ),
    )
    op.create_index(
        op.f("ix_document_embeddings_document_extraction_id"),
        "document_embeddings",
        ["document_extraction_id"],
        unique=False,
    )
    op.create_table(
        "page_embeddings",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("document_embedding_id", sa.BigInteger(), nullable=False),
        sa.Column("document_page_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.Vector(EMBEDDING_DIMENSIONS),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_embedding_id"],
            ["document_embeddings.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_page_id"],
            ["document_pages.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_embedding_id",
            "document_page_id",
            name="uq_page_embeddings_embedding_page",
        ),
    )
    op.create_index(
        op.f("ix_page_embeddings_document_embedding_id"),
        "page_embeddings",
        ["document_embedding_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_page_embeddings_document_page_id"),
        "page_embeddings",
        ["document_page_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_page_embeddings_document_page_id"), table_name="page_embeddings"
    )
    op.drop_index(
        op.f("ix_page_embeddings_document_embedding_id"), table_name="page_embeddings"
    )
    op.drop_table("page_embeddings")
    op.drop_index(
        op.f("ix_document_embeddings_document_extraction_id"),
        table_name="document_embeddings",
    )
    op.drop_table("document_embeddings")

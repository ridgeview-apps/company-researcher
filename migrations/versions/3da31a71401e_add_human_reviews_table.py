"""add human reviews table

Revision ID: 3da31a71401e
Revises: 147547a28eb8
Create Date: 2026-08-25 16:37:52.462717

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "3da31a71401e"
down_revision: str | Sequence[str] | None = "147547a28eb8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "human_reviews",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("company_number", sa.Text(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("generated_query", sa.Text(), nullable=False),
        sa.Column("claim", sa.Text(), nullable=False),
        sa.Column("claim_type", sa.Text(), nullable=False),
        sa.Column("evidence_sufficient", sa.Boolean(), nullable=False),
        sa.Column("citations", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("review_reason", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("edited_claim", sa.Text(), nullable=True),
        sa.Column("decision_note", sa.Text(), nullable=True),
        sa.Column("reviewer", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "claim_type IN ('fact', 'interpretation')",
            name="ck_human_reviews_claim_type",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'edited', 'rejected', "
            "'more_research_requested')",
            name="ck_human_reviews_status",
        ),
        sa.ForeignKeyConstraint(
            ["company_number"],
            ["companies.company_number"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_human_reviews_company_number"),
        "human_reviews",
        ["company_number"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_human_reviews_company_number"), table_name="human_reviews")
    op.drop_table("human_reviews")

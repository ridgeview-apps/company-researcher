"""add companies and filings tables

Revision ID: 5f66dd79992f
Revises: 098a19906e22
Create Date: 2026-08-19 12:25:33.658311

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "5f66dd79992f"
down_revision: str | Sequence[str] | None = "098a19906e22"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "companies",
        sa.Column("company_number", sa.Text(), nullable=False),
        sa.Column("company_name", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("company_status", sa.Text(), nullable=True),
        sa.Column("date_of_creation", sa.Date(), nullable=True),
        sa.Column("date_of_cessation", sa.Date(), nullable=True),
        sa.Column("sic_codes", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column(
            "registered_office_address",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column(
            "raw_profile", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("company_number"),
    )
    op.create_table(
        "filings",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("company_number", sa.Text(), nullable=False),
        sa.Column("transaction_id", sa.Text(), nullable=False),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("action_date", sa.Date(), nullable=True),
        sa.Column("barcode", sa.Text(), nullable=True),
        sa.Column("pages", sa.Integer(), nullable=True),
        sa.Column("paper_filed", sa.Boolean(), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column(
            "raw_filing", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["company_number"],
            ["companies.company_number"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "company_number",
            "transaction_id",
            name="uq_filings_company_number_transaction_id",
        ),
    )
    op.create_index(
        op.f("ix_filings_company_number"), "filings", ["company_number"], unique=False
    )
    op.create_index(op.f("ix_filings_date"), "filings", ["date"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_filings_date"), table_name="filings")
    op.drop_index(op.f("ix_filings_company_number"), table_name="filings")
    op.drop_table("filings")
    op.drop_table("companies")

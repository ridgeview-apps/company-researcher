from datetime import date as PyDate
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from company_researcher.db.base import Base


class Company(Base):
    """Current structured profile for one company, as last retrieved from a source."""

    __tablename__ = "companies"

    company_number: Mapped[str] = mapped_column(Text, primary_key=True)
    company_name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    company_status: Mapped[str | None] = mapped_column(Text)
    date_of_creation: Mapped[PyDate | None] = mapped_column(Date)
    date_of_cessation: Mapped[PyDate | None] = mapped_column(Date)
    sic_codes: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list
    )
    registered_office_address: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    source: Mapped[str] = mapped_column(Text, nullable=False, default="companies_house")
    raw_profile: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Filing(Base):
    """One filing-history item belonging to a company."""

    __tablename__ = "filings"
    __table_args__ = (
        UniqueConstraint(
            "company_number",
            "transaction_id",
            name="uq_filings_company_number_transaction_id",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    company_number: Mapped[str] = mapped_column(
        Text, ForeignKey("companies.company_number"), nullable=False, index=True
    )
    transaction_id: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    date: Mapped[PyDate] = mapped_column(Date, nullable=False, index=True)
    action_date: Mapped[PyDate | None] = mapped_column(Date)
    barcode: Mapped[str | None] = mapped_column(Text)
    pages: Mapped[int | None] = mapped_column(Integer)
    paper_filed: Mapped[bool | None] = mapped_column(Boolean)
    source_document_id: Mapped[str | None] = mapped_column(Text, index=True)
    document_metadata_url: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text, nullable=False, default="companies_house")
    raw_filing: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class FilingDocument(Base):
    """One immutable downloaded version of a filing's source document."""

    __tablename__ = "filing_documents"
    __table_args__ = (
        UniqueConstraint(
            "source",
            "source_document_id",
            "sha256",
            name="uq_filing_documents_source_document_sha256",
        ),
        CheckConstraint(
            "char_length(sha256) = 64",
            name="ck_filing_documents_sha256_length",
        ),
        CheckConstraint(
            "content_length > 0",
            name="ck_filing_documents_content_length_positive",
        ),
        CheckConstraint(
            "pages IS NULL OR pages >= 0",
            name="ck_filing_documents_pages_non_negative",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    filing_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("filings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source: Mapped[str] = mapped_column(Text, nullable=False, default="companies_house")
    source_document_id: Mapped[str] = mapped_column(Text, nullable=False)
    media_type: Mapped[str] = mapped_column(Text, nullable=False)
    content_length: Mapped[int] = mapped_column(BigInteger, nullable=False)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    etag: Mapped[str | None] = mapped_column(Text)
    pages: Mapped[int | None] = mapped_column(Integer)
    source_created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    first_retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    last_retrieved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

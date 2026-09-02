from datetime import date as PyDate
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
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

EMBEDDING_DIMENSIONS = 1536


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


class DocumentExtraction(Base):
    """One reproducible OCR run for an immutable filing-document version."""

    __tablename__ = "document_extractions"
    __table_args__ = (
        UniqueConstraint(
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
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed')",
            name="ck_document_extractions_status",
        ),
        CheckConstraint(
            "page_count IS NULL OR page_count >= 0",
            name="ck_document_extractions_page_count_non_negative",
        ),
        CheckConstraint(
            "total_character_count IS NULL OR total_character_count >= 0",
            name="ck_document_extractions_character_count_non_negative",
        ),
        CheckConstraint(
            "render_dpi >= 72",
            name="ck_document_extractions_render_dpi_minimum",
        ),
        CheckConstraint(
            "page_segmentation_mode BETWEEN 0 AND 13",
            name="ck_document_extractions_page_segmentation_mode",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    filing_document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("filing_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")
    extractor: Mapped[str] = mapped_column(Text, nullable=False)
    extractor_version: Mapped[str] = mapped_column(Text, nullable=False)
    renderer: Mapped[str] = mapped_column(Text, nullable=False)
    renderer_version: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(Text, nullable=False)
    render_dpi: Mapped[int] = mapped_column(Integer, nullable=False)
    page_segmentation_mode: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    total_character_count: Mapped[int | None] = mapped_column(BigInteger)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DocumentPage(Base):
    """OCR text for one page of a specific document extraction."""

    __tablename__ = "document_pages"
    __table_args__ = (
        UniqueConstraint(
            "document_extraction_id",
            "page_number",
            name="uq_document_pages_extraction_page_number",
        ),
        CheckConstraint(
            "page_number >= 1",
            name="ck_document_pages_page_number_positive",
        ),
        CheckConstraint(
            "character_count >= 0",
            name="ck_document_pages_character_count_non_negative",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_extraction_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_extractions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    character_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DocumentEmbedding(Base):
    """One reproducible embedding run over a document extraction's pages."""

    __tablename__ = "document_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "document_extraction_id",
            "provider",
            "model",
            "dimensions",
            name="uq_document_embeddings_extraction_configuration",
        ),
        CheckConstraint(
            "status IN ('running', 'succeeded', 'failed')",
            name="ck_document_embeddings_status",
        ),
        CheckConstraint(
            "page_count IS NULL OR page_count >= 0",
            name="ck_document_embeddings_page_count_non_negative",
        ),
        CheckConstraint(
            "dimensions > 0",
            name="ck_document_embeddings_dimensions_positive",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_extraction_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_extractions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, default="running")
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    page_count: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class HumanReview(Base):
    """One human-in-the-loop review of an investigation finding.

    Created by `investigation_agent.py`'s `human_review_gate` node whenever
    a finding is an interpretation (rather than a directly evidenced fact)
    or reports insufficient evidence, so a human analyst can approve, edit,
    reject, or request further research before the finding is treated as
    final. `citations` mirrors `raw_profile`/`raw_filing`'s JSONB provenance
    convention rather than a normalized table, since a citation here is
    already a closed, immutable snapshot of what the model cited at
    synthesis time - not something later queries need to join against.
    """

    __tablename__ = "human_reviews"
    __table_args__ = (
        CheckConstraint(
            "claim_type IN ('fact', 'interpretation')",
            name="ck_human_reviews_claim_type",
        ),
        CheckConstraint(
            "status IN ('pending', 'approved', 'edited', 'rejected', "
            "'more_research_requested')",
            name="ck_human_reviews_status",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    company_number: Mapped[str] = mapped_column(
        Text, ForeignKey("companies.company_number"), nullable=False, index=True
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    generated_query: Mapped[str] = mapped_column(Text, nullable=False)
    claim: Mapped[str] = mapped_column(Text, nullable=False)
    claim_type: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_sufficient: Mapped[bool] = mapped_column(Boolean, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    review_reason: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    edited_claim: Mapped[str | None] = mapped_column(Text)
    decision_note: Mapped[str | None] = mapped_column(Text)
    reviewer: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PageEmbedding(Base):
    """One page's embedding vector for a specific document embedding run."""

    __tablename__ = "page_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "document_embedding_id",
            "document_page_id",
            name="uq_page_embeddings_embedding_page",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    document_embedding_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_embeddings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_page_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("document_pages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

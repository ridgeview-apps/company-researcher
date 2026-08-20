from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.artifact_store import ArtifactStore
from company_researcher.companies_house import CompaniesHouseDocumentClient
from company_researcher.db.models import Filing, FilingDocument

SOURCE = "companies_house"
PDF_MEDIA_TYPE = "application/pdf"


class DocumentIngestionError(Exception):
    """Raised when a filing document cannot be ingested safely."""


@dataclass(frozen=True)
class DocumentIngestionResult:
    """Summary of one persisted filing-document version."""

    filing_document_id: int
    source_document_id: str
    sha256: str
    storage_key: str
    created: bool


async def ingest_filing_document(
    session: AsyncSession,
    client: CompaniesHouseDocumentClient,
    artifact_store: ArtifactStore,
    filing: Filing,
) -> DocumentIngestionResult:
    """Acquire and persist one filing's immutable PDF document version."""
    if filing.source_document_id is None:
        raise DocumentIngestionError("Filing does not reference a source document")

    metadata = await client.get_document_metadata(filing.source_document_id)
    pdf_resource = metadata.resources.get(PDF_MEDIA_TYPE)
    if pdf_resource is None:
        raise DocumentIngestionError("Filing document is not available as PDF")

    content = await client.get_document_content(filing.source_document_id)
    if len(content.content) != pdf_resource.content_length:
        raise DocumentIngestionError(
            "Downloaded document length does not match its source metadata"
        )

    stored = await artifact_store.put(content.content, extension="pdf")
    retrieved_at = datetime.now(UTC)
    existing = await session.scalar(
        select(FilingDocument).where(
            FilingDocument.source == SOURCE,
            FilingDocument.source_document_id == filing.source_document_id,
            FilingDocument.sha256 == stored.sha256,
        )
    )

    if existing is None:
        document = FilingDocument(
            filing_id=filing.id,
            source=SOURCE,
            source_document_id=filing.source_document_id,
            media_type=content.media_type,
            content_length=stored.content_length,
            sha256=stored.sha256,
            storage_key=stored.storage_key,
            etag=metadata.etag,
            pages=metadata.pages,
            source_created_at=metadata.created_at,
            source_updated_at=metadata.updated_at,
            raw_metadata=metadata.model_dump(mode="json"),
            first_retrieved_at=retrieved_at,
            last_retrieved_at=retrieved_at,
        )
        session.add(document)
        await session.commit()
        created = True
    else:
        existing.media_type = content.media_type
        existing.content_length = stored.content_length
        existing.storage_key = stored.storage_key
        existing.etag = metadata.etag
        existing.pages = metadata.pages
        existing.source_created_at = metadata.created_at
        existing.source_updated_at = metadata.updated_at
        existing.raw_metadata = metadata.model_dump(mode="json")
        existing.last_retrieved_at = retrieved_at
        await session.commit()
        document = existing
        created = False

    return DocumentIngestionResult(
        filing_document_id=document.id,
        source_document_id=document.source_document_id,
        sha256=document.sha256,
        storage_key=document.storage_key,
        created=created,
    )

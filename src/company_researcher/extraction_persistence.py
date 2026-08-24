from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.artifact_store import ArtifactStore
from company_researcher.db.models import (
    DocumentExtraction,
    DocumentPage,
    FilingDocument,
)
from company_researcher.pdf_extraction import PdfExtractionError, PdfExtractor

ExtractionOutcome = Literal["created", "retried", "reused"]


@dataclass(frozen=True)
class ExtractionPersistenceResult:
    """Summary of one persisted document extraction.

    `outcome` distinguishes three cases that a single "was it created"
    boolean conflates: 'created' (no tracking row existed before this call),
    'retried' (a row existed but its previous run had not succeeded, e.g. it
    failed — real OCR work happened just now), and 'reused' (a row already
    existed with a succeeded run — no new work happened).
    """

    document_extraction_id: int
    page_count: int
    total_character_count: int
    outcome: ExtractionOutcome


async def extract_filing_document(
    session: AsyncSession,
    artifact_store: ArtifactStore,
    extractor: PdfExtractor,
    filing_document: FilingDocument,
) -> ExtractionPersistenceResult:
    """Verify, extract, and persist one immutable filing document's pages."""
    pdf_content = await artifact_store.get(
        filing_document.storage_key,
        expected_sha256=filing_document.sha256,
    )
    configuration = extractor.configuration
    extraction = await session.scalar(
        select(DocumentExtraction).where(
            DocumentExtraction.filing_document_id == filing_document.id,
            DocumentExtraction.extractor == configuration.extractor,
            DocumentExtraction.extractor_version == configuration.extractor_version,
            DocumentExtraction.renderer == configuration.renderer,
            DocumentExtraction.renderer_version == configuration.renderer_version,
            DocumentExtraction.language == configuration.language,
            DocumentExtraction.render_dpi == configuration.render_dpi,
            DocumentExtraction.page_segmentation_mode
            == configuration.page_segmentation_mode,
        )
    )

    if extraction is not None and extraction.status == "succeeded":
        return ExtractionPersistenceResult(
            document_extraction_id=extraction.id,
            page_count=extraction.page_count or 0,
            total_character_count=extraction.total_character_count or 0,
            outcome="reused",
        )

    outcome: ExtractionOutcome = "created" if extraction is None else "retried"
    started_at = datetime.now(UTC)
    if extraction is None:
        extraction = DocumentExtraction(
            filing_document_id=filing_document.id,
            status="running",
            extractor=configuration.extractor,
            extractor_version=configuration.extractor_version,
            renderer=configuration.renderer,
            renderer_version=configuration.renderer_version,
            language=configuration.language,
            render_dpi=configuration.render_dpi,
            page_segmentation_mode=configuration.page_segmentation_mode,
            started_at=started_at,
        )
        session.add(extraction)
    else:
        await session.execute(
            delete(DocumentPage).where(
                DocumentPage.document_extraction_id == extraction.id
            )
        )
        extraction.status = "running"
        extraction.page_count = None
        extraction.total_character_count = None
        extraction.started_at = started_at
        extraction.completed_at = None
        extraction.error_message = None
    await session.commit()

    try:
        result = await extractor.extract(pdf_content)
    except PdfExtractionError as error:
        extraction.status = "failed"
        extraction.completed_at = datetime.now(UTC)
        extraction.error_message = str(error)
        await session.commit()
        raise

    pages = [
        DocumentPage(
            document_extraction_id=extraction.id,
            page_number=page.page_number,
            text=page.text,
            character_count=page.character_count,
        )
        for page in result.pages
    ]
    session.add_all(pages)
    extraction.status = "succeeded"
    extraction.page_count = len(pages)
    extraction.total_character_count = sum(page.character_count for page in pages)
    extraction.completed_at = datetime.now(UTC)
    extraction.error_message = None
    await session.commit()

    return ExtractionPersistenceResult(
        document_extraction_id=extraction.id,
        page_count=extraction.page_count,
        total_character_count=extraction.total_character_count,
        outcome=outcome,
    )

from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.artifact_store import (
    ArtifactIntegrityError,
    LocalArtifactStore,
)
from company_researcher.config import Settings
from company_researcher.db.models import (
    Company,
    DocumentExtraction,
    DocumentPage,
    Filing,
    FilingDocument,
)
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.extraction_persistence import extract_filing_document
from company_researcher.pdf_extraction import (
    ExtractedPage,
    PdfExtractionConfiguration,
    PdfExtractionError,
    PdfExtractionResult,
)

TEST_COMPANY_NUMBER = "TE000005"


class FakeExtractor:
    def __init__(self, *, fail: bool = False) -> None:
        self.call_count = 0
        self.fail = fail
        self.configuration = PdfExtractionConfiguration(
            extractor="fake-ocr",
            extractor_version="1.0",
            renderer="fake-renderer",
            renderer_version="1.0",
            language="eng",
            render_dpi=300,
            page_segmentation_mode=3,
        )

    async def extract(self, pdf_content: bytes) -> PdfExtractionResult:
        self.call_count += 1
        if self.fail:
            raise PdfExtractionError("Controlled OCR failure")
        pages = [
            ExtractedPage(page_number=1, text="First page", character_count=10),
            ExtractedPage(page_number=2, text="Second page", character_count=11),
        ]
        return PdfExtractionResult(
            pages=pages,
            extractor=self.configuration.extractor,
            extractor_version=self.configuration.extractor_version,
            renderer=self.configuration.renderer,
            renderer_version=self.configuration.renderer_version,
            language=self.configuration.language,
            render_dpi=self.configuration.render_dpi,
            page_segmentation_mode=self.configuration.page_segmentation_mode,
        )


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_database_engine(Settings(_env_file=None))  # type: ignore[call-arg]
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as db_session:
            yield db_session
    finally:
        async with session_factory() as cleanup_session:
            await cleanup_session.execute(
                delete(Filing).where(Filing.company_number == TEST_COMPANY_NUMBER)
            )
            await cleanup_session.execute(
                delete(Company).where(Company.company_number == TEST_COMPANY_NUMBER)
            )
            await cleanup_session.commit()
        await engine.dispose()


async def _create_document(
    session: AsyncSession,
    store: LocalArtifactStore,
) -> FilingDocument:
    now = datetime.now(UTC)
    company = Company(
        company_number=TEST_COMPANY_NUMBER,
        company_name="EXTRACTION PERSISTENCE TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=now,
    )
    filing = Filing(
        company_number=TEST_COMPANY_NUMBER,
        transaction_id="extraction-persistence-transaction",
        category="accounts",
        type="AA",
        description="accounts",
        date=date(2026, 1, 1),
        raw_filing={},
        retrieved_at=now,
    )
    stored = await store.put(b"%PDF-1.7 test content", extension="pdf")
    session.add_all([company, filing])
    await session.flush()
    document = FilingDocument(
        filing_id=filing.id,
        source_document_id="extraction-persistence-document",
        media_type="application/pdf",
        content_length=stored.content_length,
        sha256=stored.sha256,
        storage_key=stored.storage_key,
        source_created_at=now,
        raw_metadata={},
        first_retrieved_at=now,
        last_retrieved_at=now,
    )
    session.add(document)
    await session.commit()
    return document


@pytest.mark.asyncio
async def test_extract_filing_document_persists_pages_and_provenance(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    document = await _create_document(session, store)
    extractor = FakeExtractor()

    result = await extract_filing_document(session, store, extractor, document)
    extraction = await session.get(DocumentExtraction, result.document_extraction_id)
    pages = list(
        await session.scalars(
            select(DocumentPage)
            .where(DocumentPage.document_extraction_id == result.document_extraction_id)
            .order_by(DocumentPage.page_number)
        )
    )

    assert result.created is True
    assert result.page_count == 2
    assert result.total_character_count == 21
    assert extraction is not None
    assert extraction.status == "succeeded"
    assert [page.text for page in pages] == ["First page", "Second page"]


@pytest.mark.asyncio
async def test_extract_filing_document_skips_succeeded_configuration(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    document = await _create_document(session, store)
    extractor = FakeExtractor()

    first = await extract_filing_document(session, store, extractor, document)
    second = await extract_filing_document(session, store, extractor, document)
    page_count = await session.scalar(
        select(func.count())
        .select_from(DocumentPage)
        .where(DocumentPage.document_extraction_id == first.document_extraction_id)
    )

    assert first.created is True
    assert second.created is False
    assert extractor.call_count == 1
    assert page_count == 2


@pytest.mark.asyncio
async def test_extract_filing_document_records_controlled_failure(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    document = await _create_document(session, store)

    with pytest.raises(PdfExtractionError, match="Controlled OCR failure"):
        await extract_filing_document(
            session, store, FakeExtractor(fail=True), document
        )

    extraction = await session.scalar(
        select(DocumentExtraction).where(
            DocumentExtraction.filing_document_id == document.id
        )
    )
    assert extraction is not None
    assert extraction.status == "failed"
    assert extraction.error_message == "Controlled OCR failure"
    assert extraction.completed_at is not None


@pytest.mark.asyncio
async def test_extract_filing_document_rejects_corrupt_artifact_before_run(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    document = await _create_document(session, store)
    (tmp_path / document.storage_key).write_bytes(b"corrupt")

    with pytest.raises(ArtifactIntegrityError):
        await extract_filing_document(session, store, FakeExtractor(), document)

    extraction_count = await session.scalar(
        select(func.count())
        .select_from(DocumentExtraction)
        .where(DocumentExtraction.filing_document_id == document.id)
    )
    assert extraction_count == 0

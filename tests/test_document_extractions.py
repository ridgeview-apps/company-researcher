from collections.abc import AsyncIterator
from datetime import UTC, date, datetime

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.config import Settings
from company_researcher.db.models import (
    Company,
    DocumentExtraction,
    DocumentPage,
    Filing,
    FilingDocument,
)
from company_researcher.db.session import create_database_engine, create_session_factory

TEST_COMPANY_NUMBER = "TE000004"


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


async def _create_filing_document(session: AsyncSession) -> FilingDocument:
    now = datetime.now(UTC)
    company = Company(
        company_number=TEST_COMPANY_NUMBER,
        company_name="EXTRACTION SCHEMA TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=now,
    )
    filing = Filing(
        company_number=TEST_COMPANY_NUMBER,
        transaction_id="extraction-schema-transaction",
        category="accounts",
        type="AA",
        description="accounts",
        date=date(2026, 1, 1),
        raw_filing={},
        retrieved_at=now,
    )
    session.add_all([company, filing])
    await session.flush()
    document = FilingDocument(
        filing_id=filing.id,
        source_document_id="extraction-schema-document",
        media_type="application/pdf",
        content_length=1234,
        sha256="a" * 64,
        storage_key="sha256/test.pdf",
        source_created_at=now,
        raw_metadata={},
        first_retrieved_at=now,
        last_retrieved_at=now,
    )
    session.add(document)
    await session.flush()
    return document


def _extraction(filing_document_id: int) -> DocumentExtraction:
    return DocumentExtraction(
        filing_document_id=filing_document_id,
        extractor="tesseract",
        extractor_version="5.5.3",
        renderer="pypdfium2",
        renderer_version="5.13.0",
        language="eng",
        render_dpi=300,
        page_segmentation_mode=3,
        started_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_document_extraction_persists_page_lineage(
    session: AsyncSession,
) -> None:
    document = await _create_filing_document(session)
    extraction = _extraction(document.id)
    session.add(extraction)
    await session.flush()
    page = DocumentPage(
        document_extraction_id=extraction.id,
        page_number=1,
        text="Extracted evidence",
        character_count=18,
    )
    session.add(page)
    await session.commit()

    persisted = await session.get(DocumentPage, page.id)

    assert persisted is not None
    assert persisted.document_extraction_id == extraction.id
    assert persisted.page_number == 1
    assert persisted.text == "Extracted evidence"


@pytest.mark.asyncio
async def test_document_extraction_rejects_duplicate_configuration(
    session: AsyncSession,
) -> None:
    document = await _create_filing_document(session)
    session.add_all([_extraction(document.id), _extraction(document.id)])

    with pytest.raises(IntegrityError):
        await session.commit()

    await session.rollback()


@pytest.mark.asyncio
async def test_document_extraction_rejects_duplicate_page_number(
    session: AsyncSession,
) -> None:
    document = await _create_filing_document(session)
    extraction = _extraction(document.id)
    session.add(extraction)
    await session.flush()
    session.add_all(
        [
            DocumentPage(
                document_extraction_id=extraction.id,
                page_number=1,
                text="First",
                character_count=5,
            ),
            DocumentPage(
                document_extraction_id=extraction.id,
                page_number=1,
                text="Duplicate",
                character_count=9,
            ),
        ]
    )

    with pytest.raises(IntegrityError):
        await session.commit()

    await session.rollback()


@pytest.mark.asyncio
async def test_deleting_filing_document_cascades_extraction_and_pages(
    session: AsyncSession,
) -> None:
    document = await _create_filing_document(session)
    extraction = _extraction(document.id)
    session.add(extraction)
    await session.flush()
    session.add(
        DocumentPage(
            document_extraction_id=extraction.id,
            page_number=1,
            text="Evidence",
            character_count=8,
        )
    )
    await session.commit()

    await session.delete(document)
    await session.commit()

    assert (
        await session.scalar(
            select(DocumentExtraction).where(DocumentExtraction.id == extraction.id)
        )
        is None
    )
    assert (
        await session.scalar(
            select(DocumentPage).where(
                DocumentPage.document_extraction_id == extraction.id
            )
        )
        is None
    )

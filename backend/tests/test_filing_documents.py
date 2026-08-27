from collections.abc import AsyncIterator
from datetime import UTC, date, datetime

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.config import Settings
from company_researcher.db.models import Company, Filing, FilingDocument
from company_researcher.db.session import create_database_engine, create_session_factory

TEST_COMPANY_NUMBER = "TE000002"
TEST_TRANSACTION_ID = "document-schema-transaction"


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_database_engine(Settings(_env_file=None))  # type: ignore[call-arg]
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as db_session:
            yield db_session
    finally:
        async with session_factory() as cleanup_session:
            filing_ids = [
                filing_id
                for (filing_id,) in (
                    await cleanup_session.execute(
                        Filing.__table__.select()
                        .with_only_columns(Filing.id)
                        .where(Filing.company_number == TEST_COMPANY_NUMBER)
                    )
                )
            ]
            if filing_ids:
                await cleanup_session.execute(
                    delete(FilingDocument).where(
                        FilingDocument.filing_id.in_(filing_ids)
                    )
                )
            await cleanup_session.execute(
                delete(Filing).where(Filing.company_number == TEST_COMPANY_NUMBER)
            )
            await cleanup_session.execute(
                delete(Company).where(Company.company_number == TEST_COMPANY_NUMBER)
            )
            await cleanup_session.commit()
        await engine.dispose()


async def _create_filing(session: AsyncSession) -> Filing:
    retrieved_at = datetime.now(UTC)
    company = Company(
        company_number=TEST_COMPANY_NUMBER,
        company_name="DOCUMENT SCHEMA TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=retrieved_at,
    )
    filing = Filing(
        company_number=TEST_COMPANY_NUMBER,
        transaction_id=TEST_TRANSACTION_ID,
        category="accounts",
        type="AA",
        description="accounts",
        date=date(2026, 1, 1),
        raw_filing={},
        retrieved_at=retrieved_at,
    )
    session.add_all([company, filing])
    await session.flush()
    return filing


def _document(filing_id: int, *, checksum: str) -> FilingDocument:
    retrieved_at = datetime.now(UTC)
    return FilingDocument(
        filing_id=filing_id,
        source_document_id="document-123",
        media_type="application/pdf",
        content_length=1234,
        sha256=checksum,
        storage_key=f"sha256/{checksum}.pdf",
        pages=10,
        source_created_at=retrieved_at,
        raw_metadata={},
        first_retrieved_at=retrieved_at,
        last_retrieved_at=retrieved_at,
    )


@pytest.mark.asyncio
async def test_filing_document_persists_source_and_artifact_provenance(
    session: AsyncSession,
) -> None:
    filing = await _create_filing(session)
    document = _document(filing.id, checksum="a" * 64)
    session.add(document)
    await session.commit()

    persisted = await session.get(FilingDocument, document.id)

    assert persisted is not None
    assert persisted.filing_id == filing.id
    assert persisted.source == "companies_house"
    assert persisted.source_document_id == "document-123"
    assert persisted.sha256 == "a" * 64


@pytest.mark.asyncio
async def test_filing_document_identity_allows_new_content_version(
    session: AsyncSession,
) -> None:
    filing = await _create_filing(session)
    session.add_all(
        [
            _document(filing.id, checksum="a" * 64),
            _document(filing.id, checksum="b" * 64),
        ]
    )
    await session.commit()


@pytest.mark.asyncio
async def test_filing_document_identity_rejects_duplicate_content_version(
    session: AsyncSession,
) -> None:
    filing = await _create_filing(session)
    session.add_all(
        [
            _document(filing.id, checksum="a" * 64),
            _document(filing.id, checksum="a" * 64),
        ]
    )

    with pytest.raises(IntegrityError):
        await session.commit()

    await session.rollback()

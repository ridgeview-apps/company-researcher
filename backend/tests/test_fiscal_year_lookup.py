from collections.abc import AsyncIterator
from datetime import UTC, date, datetime

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.config import Settings
from company_researcher.db.models import (
    Company,
    DocumentExtraction,
    Filing,
    FilingDocument,
)
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.fiscal_year_lookup import (
    document_extraction_ids_for_fiscal_year,
)

TEST_COMPANY_NUMBER = "TE000009"
OTHER_TEST_COMPANY_NUMBER = "TE000011"


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_database_engine(Settings(_env_file=None))  # type: ignore[call-arg]
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as db_session:
            yield db_session
    finally:
        async with session_factory() as cleanup_session:
            for company_number in (TEST_COMPANY_NUMBER, OTHER_TEST_COMPANY_NUMBER):
                await cleanup_session.execute(
                    delete(Filing).where(Filing.company_number == company_number)
                )
                await cleanup_session.execute(
                    delete(Company).where(Company.company_number == company_number)
                )
            await cleanup_session.commit()
        await engine.dispose()


@pytest_asyncio.fixture
async def company(session: AsyncSession) -> Company:
    company = Company(
        company_number=TEST_COMPANY_NUMBER,
        company_name="FISCAL YEAR LOOKUP TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=datetime.now(UTC),
    )
    session.add(company)
    await session.commit()
    return company


@pytest_asyncio.fixture
async def other_company(session: AsyncSession) -> Company:
    company = Company(
        company_number=OTHER_TEST_COMPANY_NUMBER,
        company_name="OTHER FISCAL YEAR LOOKUP TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=datetime.now(UTC),
    )
    session.add(company)
    await session.commit()
    return company


async def _create_extraction(
    session: AsyncSession,
    transaction_id: str,
    made_up_date: str,
    *,
    company_number: str = TEST_COMPANY_NUMBER,
) -> DocumentExtraction:
    now = datetime.now(UTC)
    filing = Filing(
        company_number=company_number,
        transaction_id=transaction_id,
        category="accounts",
        type="AA",
        description="accounts",
        date=date(2026, 1, 1),
        raw_filing={"description_values": {"made_up_date": made_up_date}},
        retrieved_at=now,
    )
    session.add(filing)
    await session.flush()
    document = FilingDocument(
        filing_id=filing.id,
        source_document_id=f"{transaction_id}-document",
        media_type="application/pdf",
        content_length=1234,
        sha256=f"{abs(hash(transaction_id)):064x}"[:64],
        storage_key="sha256/test.pdf",
        source_created_at=now,
        raw_metadata={},
        first_retrieved_at=now,
        last_retrieved_at=now,
    )
    session.add(document)
    await session.flush()
    extraction = DocumentExtraction(
        filing_document_id=document.id,
        status="succeeded",
        extractor="tesseract",
        extractor_version="5.5.3",
        renderer="pypdfium2",
        renderer_version="5.13.0",
        language="eng",
        render_dpi=300,
        page_segmentation_mode=3,
        started_at=now,
    )
    session.add(extraction)
    await session.commit()
    return extraction


@pytest.mark.asyncio
async def test_document_extraction_ids_for_fiscal_year_matches_the_accounting_period(
    session: AsyncSession, company: Company
) -> None:
    correct = await _create_extraction(
        session, "fiscal-year-lookup-fy2023", "2023-07-31"
    )
    await _create_extraction(session, "fiscal-year-lookup-fy2022", "2022-07-31")

    matched_ids = await document_extraction_ids_for_fiscal_year(session, "2023")

    assert correct.id in matched_ids


@pytest.mark.asyncio
async def test_document_extraction_ids_for_fiscal_year_excludes_other_years(
    session: AsyncSession, company: Company
) -> None:
    correct = await _create_extraction(
        session, "fiscal-year-lookup-fy2023b", "2023-07-31"
    )
    amended_prior_year = await _create_extraction(
        session, "fiscal-year-lookup-amended-fy2022", "2022-07-31"
    )

    matched_ids = await document_extraction_ids_for_fiscal_year(session, "2023")

    assert correct.id in matched_ids
    assert amended_prior_year.id not in matched_ids


@pytest.mark.asyncio
async def test_document_extraction_ids_for_fiscal_year_returns_empty_for_unknown_year(
    session: AsyncSession, company: Company
) -> None:
    await _create_extraction(session, "fiscal-year-lookup-fy2023c", "2023-07-31")

    matched_ids = await document_extraction_ids_for_fiscal_year(session, "1999")

    assert matched_ids == []


@pytest.mark.asyncio
async def test_document_extraction_ids_for_fiscal_year_scopes_by_company_number(
    session: AsyncSession, company: Company, other_company: Company
) -> None:
    """Regression test for a real observed failure: without company scoping, a
    genuinely absent year for one company reads as "present" the moment any
    other company in the shared database happens to have a filing whose own
    accounting period matches the same year -- exactly what happened when a
    real Gymshark FY2024 accounts filing (unrelated to the company under
    test) leaked into an unscoped lookup and zeroed out an otherwise
    answerable question's retrieval. company_number must restrict the match
    to the target company's own filings.
    """
    other_companys_filing = await _create_extraction(
        session,
        "fiscal-year-lookup-other-company-fy2023",
        "2023-07-31",
        company_number=OTHER_TEST_COMPANY_NUMBER,
    )

    matched_ids = await document_extraction_ids_for_fiscal_year(
        session, "2023", company_number=TEST_COMPANY_NUMBER
    )

    assert other_companys_filing.id not in matched_ids
    assert matched_ids == []

    unscoped_matched_ids = await document_extraction_ids_for_fiscal_year(
        session, "2023", company_number=OTHER_TEST_COMPANY_NUMBER
    )

    assert other_companys_filing.id in unscoped_matched_ids

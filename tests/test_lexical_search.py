from collections.abc import AsyncIterator
from datetime import UTC, date, datetime

import pytest
import pytest_asyncio
from sqlalchemy import delete, select
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
from company_researcher.lexical_search import search_pages, text_matches_query

TEST_COMPANY_NUMBER = "TE000006"
OTHER_COMPANY_NUMBER = "TE000007"


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_database_engine(Settings(_env_file=None))  # type: ignore[call-arg]
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as db_session:
            yield db_session
    finally:
        async with session_factory() as cleanup_session:
            for company_number in (TEST_COMPANY_NUMBER, OTHER_COMPANY_NUMBER):
                await cleanup_session.execute(
                    delete(Filing).where(Filing.company_number == company_number)
                )
                await cleanup_session.execute(
                    delete(Company).where(Company.company_number == company_number)
                )
            await cleanup_session.commit()
        await engine.dispose()


async def _create_pages(
    session: AsyncSession,
    texts: list[str],
    *,
    transaction_id: str = "lexical-search-transaction",
    company_number: str = TEST_COMPANY_NUMBER,
    filing_date: date = date(2026, 1, 1),
) -> DocumentExtraction:
    now = datetime.now(UTC)
    company_statement = select(Company).where(Company.company_number == company_number)
    company = (await session.execute(company_statement)).scalar_one_or_none()
    if company is None:
        company = Company(
            company_number=company_number,
            company_name="LEXICAL SEARCH TEST LIMITED",
            type="ltd",
            sic_codes=[],
            raw_profile={},
            retrieved_at=now,
        )
        session.add(company)
    filing = Filing(
        company_number=company_number,
        transaction_id=transaction_id,
        category="accounts",
        type="AA",
        description="accounts",
        date=filing_date,
        raw_filing={},
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
    await session.flush()
    session.add_all(
        [
            DocumentPage(
                document_extraction_id=extraction.id,
                page_number=page_number,
                text=text,
                character_count=len(text),
            )
            for page_number, text in enumerate(texts, start=1)
        ]
    )
    await session.commit()
    return extraction


@pytest.mark.asyncio
async def test_search_pages_ranks_matching_page_first(session: AsyncSession) -> None:
    extraction = await _create_pages(
        session,
        [
            "Zephyrion auditorial exemplar report on this page.",
            "Unrelated qwibble notes about property.",
        ],
    )

    # A generous limit is required because the persisted Gymshark evaluation
    # corpus lives in the same database and would otherwise crowd out these
    # rows for common financial vocabulary; the query terms here are rare
    # enough that only this test's own pages can match at all.
    matches = await search_pages(session, "zephyrion report", limit=500)
    matching = [m for m in matches if m.document_extraction_id == extraction.id]

    assert matching[0].page_number == 1


@pytest.mark.asyncio
async def test_search_pages_excludes_non_matching_pages(session: AsyncSession) -> None:
    extraction = await _create_pages(
        session,
        [
            "Zephyrion turnover was materially higher than the prior period.",
            "Unrelated qwibble notes about property.",
        ],
    )

    matches = await search_pages(session, "zephyrion", limit=500)
    matched_pages = {
        m.page_number for m in matches if m.document_extraction_id == extraction.id
    }

    assert matched_pages == {1}


@pytest.mark.asyncio
async def test_search_pages_or_combines_multi_word_queries(
    session: AsyncSession,
) -> None:
    extraction = await _create_pages(
        session,
        ["Zephyrion turnover increased during the financial year under review."],
    )

    matches = await search_pages(
        session,
        "What was the zephyrion company's turnover for the most recent fiscal year?",
        limit=500,
    )

    assert any(
        m.document_extraction_id == extraction.id and m.page_number == 1
        for m in matches
    )


@pytest.mark.asyncio
async def test_search_pages_respects_limit(session: AsyncSession) -> None:
    await _create_pages(
        session,
        [f"Zephyrion figure number {n} appears on this page." for n in range(5)],
    )

    matches = await search_pages(session, "zephyrion", limit=2)

    assert len(matches) == 2


@pytest.mark.asyncio
async def test_search_pages_restricts_to_given_document_extraction_ids(
    session: AsyncSession,
) -> None:
    allowed = await _create_pages(
        session,
        ["Zephyrion turnover figure for the allowed extraction."],
        transaction_id="lexical-search-allowed",
    )
    excluded = await _create_pages(
        session,
        ["Zephyrion turnover figure for the excluded extraction."],
        transaction_id="lexical-search-excluded",
    )

    matches = await search_pages(
        session, "zephyrion turnover", limit=500, document_extraction_ids=[allowed.id]
    )
    matched_extraction_ids = {m.document_extraction_id for m in matches}

    assert allowed.id in matched_extraction_ids
    assert excluded.id not in matched_extraction_ids


@pytest.mark.asyncio
async def test_search_pages_restricts_to_given_company_number(
    session: AsyncSession,
) -> None:
    allowed = await _create_pages(
        session,
        ["Zephyrion turnover figure for the allowed company."],
        transaction_id="lexical-search-company-allowed",
        company_number=TEST_COMPANY_NUMBER,
    )
    excluded = await _create_pages(
        session,
        ["Zephyrion turnover figure for the excluded company."],
        transaction_id="lexical-search-company-excluded",
        company_number=OTHER_COMPANY_NUMBER,
    )

    matches = await search_pages(
        session, "zephyrion turnover", limit=500, company_number=TEST_COMPANY_NUMBER
    )
    matched_extraction_ids = {m.document_extraction_id for m in matches}

    assert allowed.id in matched_extraction_ids
    assert excluded.id not in matched_extraction_ids


@pytest.mark.asyncio
async def test_search_pages_restricts_to_filings_on_or_before_as_of_date(
    session: AsyncSession,
) -> None:
    before_cutoff = await _create_pages(
        session,
        ["Zephyrion turnover figure from the original filing."],
        transaction_id="lexical-search-as-of-before",
        filing_date=date(2023, 4, 22),
    )
    after_cutoff = await _create_pages(
        session,
        ["Zephyrion turnover figure from the amended filing."],
        transaction_id="lexical-search-as-of-after",
        filing_date=date(2023, 11, 23),
    )

    matches = await search_pages(
        session, "zephyrion turnover", limit=500, as_of_date=date(2023, 9, 1)
    )
    matched_extraction_ids = {m.document_extraction_id for m in matches}

    assert before_cutoff.id in matched_extraction_ids
    assert after_cutoff.id not in matched_extraction_ids


@pytest.mark.asyncio
async def test_search_pages_as_of_date_on_the_filing_date_itself_is_included(
    session: AsyncSession,
) -> None:
    extraction = await _create_pages(
        session,
        ["Zephyrion turnover figure filed exactly on the cutoff date."],
        transaction_id="lexical-search-as-of-exact",
        filing_date=date(2023, 4, 22),
    )

    matches = await search_pages(
        session, "zephyrion turnover", limit=500, as_of_date=date(2023, 4, 22)
    )
    matched_extraction_ids = {m.document_extraction_id for m in matches}

    assert extraction.id in matched_extraction_ids


@pytest.mark.asyncio
async def test_search_pages_as_of_date_returns_nothing_rather_than_falling_back(
    session: AsyncSession,
) -> None:
    await _create_pages(
        session,
        ["Zephyrion turnover figure from a filing after the cutoff."],
        transaction_id="lexical-search-as-of-none-qualify",
        filing_date=date(2023, 4, 22),
    )

    matches = await search_pages(
        session, "zephyrion turnover", limit=500, as_of_date=date(2020, 1, 1)
    )

    assert matches == []


@pytest.mark.asyncio
async def test_search_pages_composes_as_of_date_with_document_extraction_ids(
    session: AsyncSession,
) -> None:
    fiscal_year_match_before_cutoff = await _create_pages(
        session,
        ["Zephyrion turnover figure, same fiscal year, filed before the cutoff."],
        transaction_id="lexical-search-compose-before",
        filing_date=date(2023, 4, 22),
    )
    fiscal_year_match_after_cutoff = await _create_pages(
        session,
        ["Zephyrion turnover figure, same fiscal year, filed after the cutoff."],
        transaction_id="lexical-search-compose-after",
        filing_date=date(2023, 11, 23),
    )
    other_fiscal_year_before_cutoff = await _create_pages(
        session,
        ["Zephyrion turnover figure, different fiscal year, filed before the cutoff."],
        transaction_id="lexical-search-compose-other-year",
        filing_date=date(2023, 1, 1),
    )

    matches = await search_pages(
        session,
        "zephyrion turnover",
        limit=500,
        document_extraction_ids=[
            fiscal_year_match_before_cutoff.id,
            fiscal_year_match_after_cutoff.id,
        ],
        as_of_date=date(2023, 9, 1),
    )
    matched_extraction_ids = {m.document_extraction_id for m in matches}

    assert matched_extraction_ids == {fiscal_year_match_before_cutoff.id}
    assert fiscal_year_match_after_cutoff.id not in matched_extraction_ids
    assert other_fiscal_year_before_cutoff.id not in matched_extraction_ids


@pytest.mark.asyncio
async def test_text_matches_query_true_when_a_term_overlaps(
    session: AsyncSession,
) -> None:
    matched = await text_matches_query(
        session, "the going concern basis is appropriate", "going concern"
    )

    assert matched is True


@pytest.mark.asyncio
async def test_text_matches_query_stems_word_forms(session: AsyncSession) -> None:
    """A query built from a noun form ("resignations") must still match text using a different, related word form ("resigned") - the same stemming `search_pages` already relies on for retrieval."""
    matched = await text_matches_query(
        session, "the director resigned in March", "resignations"
    )

    assert matched is True


@pytest.mark.asyncio
async def test_text_matches_query_false_when_genuinely_unrelated(
    session: AsyncSession,
) -> None:
    matched = await text_matches_query(
        session,
        "the company continued to trade in retail distribution",
        "fraud investigation",
    )

    assert matched is False

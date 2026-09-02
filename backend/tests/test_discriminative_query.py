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
    DocumentPage,
    Filing,
    FilingDocument,
)
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.discriminative_query import derive_discriminative_query

TEST_COMPANY_NUMBER = "TE000008"


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


async def _create_pages(session: AsyncSession, texts: list[str]) -> None:
    now = datetime.now(UTC)
    company = Company(
        company_number=TEST_COMPANY_NUMBER,
        company_name="DISCRIMINATIVE QUERY TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=now,
    )
    filing = Filing(
        company_number=TEST_COMPANY_NUMBER,
        transaction_id="discriminative-query-transaction",
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
        source_document_id="discriminative-query-document",
        media_type="application/pdf",
        content_length=1234,
        sha256="c" * 64,
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


@pytest.mark.asyncio
async def test_derive_discriminative_query_prefers_rarer_terms(
    session: AsyncSession,
) -> None:
    await _create_pages(
        session,
        [
            "Flimbernut quombat report for review.",
            "Quombat details about accounts.",
            "Quombat summary for the period.",
        ],
    )

    query = await derive_discriminative_query(
        session, "What is the flimbernut quombat figure?", max_terms=1
    )

    assert query == "flimbernut"


@pytest.mark.asyncio
async def test_derive_discriminative_query_drops_terms_absent_from_the_corpus(
    session: AsyncSession,
) -> None:
    # A generous max_terms is required to prove absent terms are dropped
    # rather than merely outranked; both invented words are otherwise
    # unlikely to appear anywhere in the persisted Gymshark corpus sharing
    # this database (see test_lexical_search.py for the same caveat).
    await _create_pages(session, ["Flimbernut appears on this page only."])

    query = await derive_discriminative_query(
        session, "What is the flimbernut zibbleplex?", max_terms=4
    )

    assert query == "flimbernut"


@pytest.mark.asyncio
async def test_derive_discriminative_query_returns_empty_string_when_nothing_matches(
    session: AsyncSession,
) -> None:
    await _create_pages(session, ["Unrelated quombat content."])

    query = await derive_discriminative_query(session, "What is the zibbleplex?")

    assert query == ""

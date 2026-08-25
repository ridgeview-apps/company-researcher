from collections.abc import AsyncIterator, Sequence
from datetime import UTC, date, datetime
from typing import TypeVar, cast

import pytest
import pytest_asyncio
from pydantic import BaseModel
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
from company_researcher.investigation_agent import (
    Citation,
    Finding,
    InvestigationAgentError,
    _force_unambiguous_fiscal_year,
    investigate,
)
from company_researcher.llm_client import ChatMessage

TEST_COMPANY_NUMBER = "TE000008"

_StructuredResponse = TypeVar("_StructuredResponse", bound=BaseModel)


class FakeChatClient:
    """Returns a fixed query for `complete` and a fixed finding for `complete_structured`."""

    def __init__(self, *, query: str, finding: Finding) -> None:
        self._query = query
        self._finding = finding
        self.complete_calls: list[Sequence[ChatMessage]] = []
        self.complete_structured_calls: list[Sequence[ChatMessage]] = []

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        self.complete_calls.append(messages)
        return self._query

    async def complete_structured(
        self, messages: Sequence[ChatMessage], response_model: type[_StructuredResponse]
    ) -> _StructuredResponse:
        self.complete_structured_calls.append(messages)
        return cast(_StructuredResponse, self._finding)


def test_force_unambiguous_fiscal_year_appends_a_missing_single_year() -> None:
    query = _force_unambiguous_fiscal_year(
        "going concern committed facility", "What was the position in FY2023?"
    )

    assert query == "going concern committed facility 2023"


def test_force_unambiguous_fiscal_year_does_not_duplicate_an_already_present_year() -> (
    None
):
    query = _force_unambiguous_fiscal_year(
        "Gymshark turnover 2025", "What was turnover for FY2025?"
    )

    assert query == "Gymshark turnover 2025"


def test_force_unambiguous_fiscal_year_leaves_range_questions_unchanged() -> None:
    query = _force_unambiguous_fiscal_year(
        "turnover cost of sales gross profit",
        "How did turnover change year-over-year from FY2021 through FY2025?",
    )

    assert query == "turnover cost of sales gross profit"


def test_force_unambiguous_fiscal_year_leaves_yearless_questions_unchanged() -> None:
    query = _force_unambiguous_fiscal_year(
        "directors secretary registered office", "Who were the directors?"
    )

    assert query == "directors secretary registered office"


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


@pytest_asyncio.fixture
async def company(session: AsyncSession) -> Company:
    company = Company(
        company_number=TEST_COMPANY_NUMBER,
        company_name="INVESTIGATION AGENT TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=datetime.now(UTC),
    )
    session.add(company)
    await session.commit()
    return company


async def _create_filing_with_pages(
    session: AsyncSession,
    transaction_id: str,
    texts: list[str],
    *,
    made_up_date: str | None = None,
) -> DocumentExtraction:
    now = datetime.now(UTC)
    raw_filing = (
        {"description_values": {"made_up_date": made_up_date}} if made_up_date else {}
    )
    filing = Filing(
        company_number=TEST_COMPANY_NUMBER,
        transaction_id=transaction_id,
        category="accounts",
        type="AA",
        description="accounts",
        date=date(2026, 1, 1),
        raw_filing=raw_filing,
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
async def test_investigate_returns_a_citation_grounded_finding(
    session: AsyncSession, company: Company
) -> None:
    # `search_pages` is not scoped by company (a pre-existing, documented
    # limitation - see README.md), so this test runs against the shared
    # development database alongside the real persisted Gymshark corpus.
    # Fixture text and the fake query must be deliberately distinctive
    # nonsense, matching test_retrieval_evaluation.py's convention, or
    # OR-combined ts_rank search will pick up unrelated real filing pages.
    extraction = await _create_filing_with_pages(
        session,
        "investigation-transaction-alpha",
        [
            "Alpha bravo charlie identifies a delta echo foxtrot as evidence.",
            "Golf hotel india unrelated content.",
        ],
    )
    expected_finding = Finding(
        claim="Alpha bravo charlie relied on a delta echo foxtrot.",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="delta echo foxtrot",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="alpha bravo charlie delta echo foxtrot", finding=expected_finding
    )

    finding = await investigate(
        session,
        chat_client,
        "What did alpha bravo charlie identify as evidence?",
    )

    assert finding == expected_finding
    assert len(chat_client.complete_calls) == 1
    assert chat_client.complete_calls[0][-1] == ChatMessage(
        role="user",
        content="What did alpha bravo charlie identify as evidence?",
    )
    assert len(chat_client.complete_structured_calls) == 1
    synthesis_prompt = chat_client.complete_structured_calls[0][-1].content
    assert "delta echo foxtrot" in synthesis_prompt
    assert f"document_extraction_id={extraction.id} page_number=1" in synthesis_prompt


@pytest.mark.asyncio
async def test_investigate_rejects_a_finding_that_cites_unretrieved_evidence(
    session: AsyncSession, company: Company
) -> None:
    extraction = await _create_filing_with_pages(
        session,
        "investigation-transaction-beta",
        ["Juliett kilo lima mentions a mike november oscar."],
    )
    hallucinated_finding = Finding(
        claim="Fabricated claim citing a page never retrieved.",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=99,
                supporting_text="does not exist",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="juliett kilo lima mike november oscar", finding=hallucinated_finding
    )

    with pytest.raises(InvestigationAgentError):
        await investigate(session, chat_client, "What did juliett kilo lima mention?")


@pytest.mark.asyncio
async def test_investigate_reports_insufficient_evidence_when_nothing_is_retrieved(
    session: AsyncSession, company: Company
) -> None:
    insufficient_finding = Finding(
        claim="The retrieved evidence does not address this question.",
        evidence_sufficient=False,
        citations=[],
    )
    # A single unbroken nonsense token: PostgreSQL's text-search parser
    # splits on underscores/punctuation, so a phrase built from ordinary
    # words (even deliberately odd ones) risks a stray token matching real
    # corpus boilerplate. One fabricated token cannot match anything.
    chat_client = FakeChatClient(query="zqxvwkploqnhfbyt", finding=insufficient_finding)

    finding = await investigate(
        session, chat_client, "What is completely unrelated to this corpus?"
    )

    assert finding.evidence_sufficient is False
    assert finding.citations == []
    synthesis_prompt = chat_client.complete_structured_calls[0][-1].content
    assert "No evidence pages were retrieved" in synthesis_prompt


@pytest.mark.asyncio
async def test_investigate_disambiguates_near_duplicate_pages_by_forced_year(
    session: AsyncSession, company: Company
) -> None:
    """Regression test for the cross-fiscal-year evidence-mixing bug (see README.md).

    Two filings share near-identical boilerplate differing only by year. The
    fake LLM's generated query omits the year, as observed in the real
    intermittent failure; `_force_unambiguous_fiscal_year` must still steer
    lexical search to the year the question actually names.
    """
    correct_extraction = await _create_filing_with_pages(
        session,
        "investigation-transaction-year-2023",
        ["Quebec romeo sierra tango whiskey xray disclosure for 2023."],
        made_up_date="2023-07-31",
    )
    await _create_filing_with_pages(
        session,
        "investigation-transaction-year-2022",
        ["Quebec romeo sierra tango whiskey xray disclosure for 2022."],
        made_up_date="2022-07-31",
    )
    expected_finding = Finding(
        claim="Quebec romeo sierra tango whiskey xray, per the 2023 filing.",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=correct_extraction.id,
                page_number=1,
                supporting_text="disclosure for 2023",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="quebec romeo sierra tango whiskey xray disclosure",
        finding=expected_finding,
    )

    finding = await investigate(
        session,
        chat_client,
        "What did quebec romeo sierra tango whiskey xray disclose in the 2023 filing?",
        context_pages=1,
    )

    assert finding == expected_finding


@pytest.mark.asyncio
async def test_investigate_excludes_a_different_fiscal_years_filing_entirely(
    session: AsyncSession, company: Company
) -> None:
    """A wrong-year filing must be excluded from evidence even if its page text
    happens to literally contain the target year (e.g. a document amended and
    signed in a later year than the accounting period it reports on -- the
    real cause of the observed leak, see README.md). Filtering must key off
    each filing's actual accounting period (`made_up_date`), not page text.
    """
    correct_extraction = await _create_filing_with_pages(
        session,
        "investigation-transaction-fy2023-real",
        ["Yankee zulu alpha beta gamma disclosure for the year ended 2023."],
        made_up_date="2023-07-31",
    )
    wrong_year_extraction = await _create_filing_with_pages(
        session,
        "investigation-transaction-fy2022-amended-2023",
        # This page's accounting period is FY2022, but it literally contains
        # "2023" too (e.g. an amendment signed in 2023) -- exactly the
        # scenario that defeats a page-text-based year filter.
        ["Yankee zulu alpha beta gamma disclosure, signed in 2023."],
        made_up_date="2022-07-31",
    )
    hallucinated_finding = Finding(
        claim="Fabricated claim citing the wrong fiscal year's filing.",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=wrong_year_extraction.id,
                page_number=1,
                supporting_text="signed in 2023",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="yankee zulu alpha beta gamma disclosure", finding=hallucinated_finding
    )

    with pytest.raises(InvestigationAgentError):
        await investigate(
            session,
            chat_client,
            "What did yankee zulu alpha beta gamma disclose in the 2023 filing?",
            context_pages=2,
        )

    # The correct FY2023 filing's page must still be reachable.
    correct_finding = Finding(
        claim="Yankee zulu alpha beta gamma, per the 2023 filing.",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=correct_extraction.id,
                page_number=1,
                supporting_text="disclosure for the year ended 2023",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="yankee zulu alpha beta gamma disclosure", finding=correct_finding
    )

    finding = await investigate(
        session,
        chat_client,
        "What did yankee zulu alpha beta gamma disclose in the 2023 filing?",
        context_pages=2,
    )

    assert finding == correct_finding

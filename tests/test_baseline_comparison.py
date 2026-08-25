from collections.abc import AsyncIterator, Sequence
from datetime import UTC, date, datetime
from typing import TypeVar, cast

import pytest
import pytest_asyncio
from pydantic import BaseModel
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.baseline_comparison import _citation_realism, compare_question
from company_researcher.config import Settings
from company_researcher.db.models import (
    Company,
    DocumentExtraction,
    DocumentPage,
    Filing,
    FilingDocument,
)
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.investigation_agent import Citation, Finding
from company_researcher.llm_client import ChatMessage, ChatUsage
from company_researcher.retrieval_evaluation import EvaluationQuestion

TEST_COMPANY_NUMBER = "TE000009"

_StructuredResponse = TypeVar("_StructuredResponse", bound=BaseModel)


class FakeComparisonChatClient:
    """Satisfies both `ChatProvider` (for `investigate()`) and `UsageAwareChatProvider` (for the baseline).

    `complete` drives `generate_query`; `complete_structured` drives
    `synthesize_finding`; `complete_structured_with_usage` drives the
    no-retrieval baseline. Kept separate from `test_investigation_agent.py`'s
    `FakeChatClient` since this one also needs the usage-aware method.
    """

    def __init__(
        self,
        *,
        query: str,
        specialized_finding: Finding,
        baseline_finding: Finding,
        baseline_usage: ChatUsage | None = None,
    ) -> None:
        self._query = query
        self._specialized_finding = specialized_finding
        self._baseline_finding = baseline_finding
        self._baseline_usage = baseline_usage
        self.complete_structured_with_usage_calls: list[Sequence[ChatMessage]] = []

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        return self._query

    async def complete_structured(
        self, messages: Sequence[ChatMessage], response_model: type[_StructuredResponse]
    ) -> _StructuredResponse:
        return cast(_StructuredResponse, self._specialized_finding)

    async def complete_structured_with_usage(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[_StructuredResponse],
    ) -> tuple[_StructuredResponse, ChatUsage | None]:
        self.complete_structured_with_usage_calls.append(messages)
        return cast(_StructuredResponse, self._baseline_finding), self._baseline_usage


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
        company_name="BASELINE COMPARISON TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=datetime.now(UTC),
    )
    session.add(company)
    await session.commit()
    return company


async def _create_filing_with_pages(
    session: AsyncSession, transaction_id: str, texts: list[str]
) -> DocumentExtraction:
    now = datetime.now(UTC)
    filing = Filing(
        company_number=TEST_COMPANY_NUMBER,
        transaction_id=transaction_id,
        category="accounts",
        type="AA",
        description="accounts",
        date=date(2026, 1, 1),
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
async def test_citation_realism_distinguishes_real_from_fabricated_pages(
    session: AsyncSession, company: Company
) -> None:
    extraction = await _create_filing_with_pages(
        session,
        "baseline-comparison-realism",
        ["Kilo lima mike disclosure on this real page."],
    )
    real_citation = Citation(
        document_extraction_id=extraction.id,
        page_number=1,
        supporting_text="disclosure on this real page",
    )
    fabricated_citation = Citation(
        document_extraction_id=999999,
        page_number=1,
        supporting_text="this page does not exist",
    )

    results = await _citation_realism(session, [real_citation, fabricated_citation])

    assert [(r.citation, r.exists) for r in results] == [
        (real_citation, True),
        (fabricated_citation, False),
    ]


@pytest.mark.asyncio
async def test_citation_realism_returns_empty_for_no_citations(
    session: AsyncSession,
) -> None:
    assert await _citation_realism(session, []) == ()


@pytest.mark.asyncio
async def test_compare_question_reports_both_findings_and_citation_realism(
    session: AsyncSession, company: Company
) -> None:
    extraction = await _create_filing_with_pages(
        session,
        "baseline-comparison-both-findings",
        ["November oscar papa disclosure of a real figure: 42."],
    )
    specialized_finding = Finding(
        claim="November oscar papa was 42.",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="disclosure of a real figure: 42",
            )
        ],
    )
    baseline_finding = Finding(
        claim="November oscar papa was probably around 40.",
        evidence_sufficient=False,
        citations=[
            Citation(
                document_extraction_id=999999,
                page_number=1,
                supporting_text="a fabricated citation with no real page",
            )
        ],
    )
    usage = ChatUsage(prompt_tokens=40, completion_tokens=15, total_tokens=55)
    chat_client = FakeComparisonChatClient(
        query="november oscar papa",
        specialized_finding=specialized_finding,
        baseline_finding=baseline_finding,
        baseline_usage=usage,
    )
    question = EvaluationQuestion(
        id="q-fake",
        text="What was november oscar papa?",
        query="november oscar papa",
        relevant_pages=(),
    )

    comparison = await compare_question(
        session,
        chat_client,
        question,
        TEST_COMPANY_NUMBER,
        "Baseline Comparison Test Limited",
    )

    assert comparison.question_id == "q-fake"
    assert comparison.specialized_finding == specialized_finding
    assert comparison.specialized_error is None
    assert comparison.baseline_finding == baseline_finding
    assert comparison.baseline_usage == usage
    assert comparison.baseline_latency_seconds >= 0
    assert comparison.specialized_latency_seconds >= 0
    assert len(comparison.baseline_citation_realism) == 1
    assert comparison.baseline_citation_realism[0].exists is False

    user_message = chat_client.complete_structured_with_usage_calls[0][-1]
    assert "Baseline Comparison Test Limited" in user_message.content
    assert "What was november oscar papa?" in user_message.content


@pytest.mark.asyncio
async def test_compare_question_catches_a_specialized_agent_citation_error(
    session: AsyncSession, company: Company
) -> None:
    """The specialized agent refusing a fabricated citation is a comparison result, not a crash."""
    extraction = await _create_filing_with_pages(
        session,
        "baseline-comparison-specialized-error",
        ["Quebec romeo sierra disclosure."],
    )
    hallucinated_specialized_finding = Finding(
        claim="Fabricated claim citing an unretrieved page.",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=99,
                supporting_text="does not exist",
            )
        ],
    )
    baseline_finding = Finding(
        claim="Unknown.", evidence_sufficient=False, citations=[]
    )
    chat_client = FakeComparisonChatClient(
        query="zqxvwkploqnhfbyt",
        specialized_finding=hallucinated_specialized_finding,
        baseline_finding=baseline_finding,
    )
    question = EvaluationQuestion(
        id="q-fake-error",
        text="What did quebec romeo sierra disclose?",
        query="quebec romeo sierra",
        relevant_pages=(),
    )

    comparison = await compare_question(
        session,
        chat_client,
        question,
        TEST_COMPANY_NUMBER,
        "Baseline Comparison Test Limited",
    )

    assert comparison.specialized_finding is None
    assert comparison.specialized_error is not None
    assert "not part of the retrieved evidence" in comparison.specialized_error

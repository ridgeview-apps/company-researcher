from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TypeVar, cast

import httpx2
import pytest
import pytest_asyncio
from pydantic import BaseModel
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.artifact_store import LocalArtifactStore
from company_researcher.baseline_agent import _BASELINE_SYSTEM_PROMPT
from company_researcher.baseline_comparison import _citation_realism, compare_question
from company_researcher.companies_house import (
    CompaniesHouseClient,
    CompaniesHouseDocumentClient,
)
from company_researcher.config import Settings
from company_researcher.db.models import (
    Company,
    DocumentExtraction,
    DocumentPage,
    Filing,
    FilingDocument,
    HumanReview,
)
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.investigation_agent import Citation, Finding
from company_researcher.llm_client import ChatMessage, ChatUsage, ToolCallTurn
from company_researcher.pdf_extraction import (
    ExtractedPage,
    PdfExtractionConfiguration,
    PdfExtractionResult,
)
from company_researcher.retrieval_evaluation import EvaluationQuestion
from company_researcher.tool_baseline_agent import _TOOL_BASELINE_SYSTEM_PROMPT

TEST_COMPANY_NUMBER = "TE000009"

_StructuredResponse = TypeVar("_StructuredResponse", bound=BaseModel)

_EMPTY_TOOL_BASELINE_FINDING = Finding(
    claim="No evidence gathered.",
    claim_type="fact",
    evidence_sufficient=False,
    citations=[],
)


class FakeExtractor:
    """A `PdfExtractor` double never actually exercised by these tests' tool-baseline path."""

    def __init__(self) -> None:
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
        return PdfExtractionResult(
            pages=[ExtractedPage(page_number=1, text="unused", character_count=6)],
            extractor=self.configuration.extractor,
            extractor_version=self.configuration.extractor_version,
            renderer=self.configuration.renderer,
            renderer_version=self.configuration.renderer_version,
            language=self.configuration.language,
            render_dpi=self.configuration.render_dpi,
            page_segmentation_mode=self.configuration.page_segmentation_mode,
        )


_Handler = Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]


def _companies_house_handler() -> _Handler:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"/company/{TEST_COMPANY_NUMBER}":
            return httpx2.Response(
                200,
                json={
                    "company_name": "Baseline Comparison Test Limited",
                    "company_number": TEST_COMPANY_NUMBER,
                    "type": "ltd",
                },
                request=request,
            )
        if request.url.path == f"/company/{TEST_COMPANY_NUMBER}/filing-history":
            return httpx2.Response(
                200,
                json={
                    "items": [],
                    "items_per_page": 100,
                    "start_index": 0,
                    "total_count": 0,
                },
                request=request,
            )
        return httpx2.Response(404, json={"error": "not found"}, request=request)

    return handler


@pytest.fixture
def companies_house_client() -> CompaniesHouseClient:
    return CompaniesHouseClient(
        api_key="test-api-key",
        base_url="https://company.example.test",
        transport=httpx2.MockTransport(_companies_house_handler()),
    )


async def _unused_document_handler(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(404, json={"error": "not found"}, request=request)


@pytest.fixture
def document_client() -> CompaniesHouseDocumentClient:
    return CompaniesHouseDocumentClient(
        api_key="test-api-key",
        base_url="https://document.example.test",
        transport=httpx2.MockTransport(_unused_document_handler),
    )


@pytest.fixture
def artifact_store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


@pytest.fixture
def extractor() -> FakeExtractor:
    return FakeExtractor()


class FakeComparisonChatClient:
    """Satisfies `FullChatProvider`, routing by which system prompt a call carries.

    `investigate_with_usage()` (query generation + synthesis),
    `answer_without_retrieval()` (the no-tool baseline), and
    `answer_with_tools()` (the tool-using baseline) all call the same
    shared client, so this fake distinguishes them by inspecting the
    system message: `_BASELINE_SYSTEM_PROMPT` and
    `_TOOL_BASELINE_SYSTEM_PROMPT` are each exact matches, everything else
    is treated as the specialized agent's call. `complete_with_tools_and_usage`
    always immediately stops the tool-calling loop (no tool calls
    requested) - these tests exercise the no-tool-baseline-vs-specialized
    comparison specifically; `test_tool_baseline_agent.py` covers the
    tool-calling loop's own mechanics.
    """

    def __init__(
        self,
        *,
        query: str,
        specialized_finding: Finding,
        baseline_finding: Finding,
        tool_baseline_finding: Finding = _EMPTY_TOOL_BASELINE_FINDING,
        baseline_usage: ChatUsage | None = None,
        specialized_usage: ChatUsage | None = None,
    ) -> None:
        self._query = query
        self._specialized_finding = specialized_finding
        self._baseline_finding = baseline_finding
        self._tool_baseline_finding = tool_baseline_finding
        self._baseline_usage = baseline_usage
        self._specialized_usage = specialized_usage
        self.complete_structured_with_usage_calls: list[Sequence[ChatMessage]] = []

    async def complete_with_usage(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[str, ChatUsage | None]:
        return self._query, self._specialized_usage

    async def complete_with_tools_and_usage(
        self, messages: Sequence[ChatMessage], tools: object
    ) -> tuple[ToolCallTurn, ChatUsage | None]:
        return ToolCallTurn(content="", tool_calls=()), None

    async def complete_structured_with_usage(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[_StructuredResponse],
    ) -> tuple[_StructuredResponse, ChatUsage | None]:
        self.complete_structured_with_usage_calls.append(messages)
        if messages[0].content == _BASELINE_SYSTEM_PROMPT:
            return cast(
                _StructuredResponse, self._baseline_finding
            ), self._baseline_usage
        if messages[0].content == _TOOL_BASELINE_SYSTEM_PROMPT:
            return cast(_StructuredResponse, self._tool_baseline_finding), None
        return cast(
            _StructuredResponse, self._specialized_finding
        ), self._specialized_usage


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
                delete(HumanReview).where(
                    HumanReview.company_number == TEST_COMPANY_NUMBER
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
    session: AsyncSession,
    company: Company,
    companies_house_client: CompaniesHouseClient,
    document_client: CompaniesHouseDocumentClient,
    artifact_store: LocalArtifactStore,
    extractor: FakeExtractor,
) -> None:
    extraction = await _create_filing_with_pages(
        session,
        "baseline-comparison-both-findings",
        ["November oscar papa disclosure of a real figure: 42."],
    )
    specialized_finding = Finding(
        claim="November oscar papa was 42.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="November oscar papa disclosure of a real figure: 42",
            )
        ],
    )
    baseline_finding = Finding(
        claim="November oscar papa was probably around 40.",
        claim_type="fact",
        evidence_sufficient=False,
        citations=[
            Citation(
                document_extraction_id=999999,
                page_number=1,
                supporting_text="a fabricated citation with no real page",
            )
        ],
    )
    baseline_usage = ChatUsage(prompt_tokens=40, completion_tokens=15, total_tokens=55)
    specialized_call_usage = ChatUsage(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    chat_client = FakeComparisonChatClient(
        query="november oscar papa",
        specialized_finding=specialized_finding,
        baseline_finding=baseline_finding,
        baseline_usage=baseline_usage,
        specialized_usage=specialized_call_usage,
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
        companies_house_client,
        document_client,
        artifact_store,
        extractor,
        question,
        TEST_COMPANY_NUMBER,
        "Baseline Comparison Test Limited",
    )

    assert comparison.question_id == "q-fake"
    assert comparison.specialized_finding == specialized_finding
    assert comparison.specialized_error is None
    # generate_query + synthesize_finding + the claim_type reclassification
    # call (see `_apply_review_integrity_checks`) each contribute
    # specialized_call_usage.
    assert comparison.specialized_usage == ChatUsage(
        prompt_tokens=30, completion_tokens=15, total_tokens=45
    )
    assert comparison.baseline_finding == baseline_finding
    assert comparison.baseline_usage == baseline_usage
    assert comparison.baseline_latency_seconds >= 0
    assert comparison.specialized_latency_seconds >= 0
    assert len(comparison.baseline_citation_realism) == 1
    assert comparison.baseline_citation_realism[0].exists is False

    user_message = chat_client.complete_structured_with_usage_calls[0][-1]
    assert "Baseline Comparison Test Limited" in user_message.content
    assert "What was november oscar papa?" in user_message.content


@pytest.mark.asyncio
async def test_compare_question_catches_a_specialized_agent_citation_error(
    session: AsyncSession,
    company: Company,
    companies_house_client: CompaniesHouseClient,
    document_client: CompaniesHouseDocumentClient,
    artifact_store: LocalArtifactStore,
    extractor: FakeExtractor,
) -> None:
    """The specialized agent refusing a fabricated citation is a comparison result, not a crash."""
    extraction = await _create_filing_with_pages(
        session,
        "baseline-comparison-specialized-error",
        ["Quebec romeo sierra disclosure."],
    )
    hallucinated_specialized_finding = Finding(
        claim="Fabricated claim citing an unretrieved page.",
        claim_type="fact",
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
        claim="Unknown.", claim_type="fact", evidence_sufficient=False, citations=[]
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
        companies_house_client,
        document_client,
        artifact_store,
        extractor,
        question,
        TEST_COMPANY_NUMBER,
        "Baseline Comparison Test Limited",
    )

    assert comparison.specialized_finding is None
    assert comparison.specialized_error is not None
    assert "not part of the retrieved evidence" in comparison.specialized_error

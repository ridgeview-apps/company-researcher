from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast

import httpx2
import pytest
import pytest_asyncio
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.artifact_store import LocalArtifactStore
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
)
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.investigation_agent import Citation, Finding
from company_researcher.llm_client import ChatMessage, ChatUsage, ToolCall, ToolCallTurn
from company_researcher.pdf_extraction import (
    ExtractedPage,
    PdfExtractionConfiguration,
    PdfExtractionResult,
)
from company_researcher.tool_baseline_agent import (
    ToolBaselineAgentError,
    ToolBaselineAnswer,
    answer_with_tools,
)

TEST_COMPANY_NUMBER = "TE000010"
OTHER_TEST_COMPANY_NUMBER = "TE000014"
_TRANSACTION_ID = "tool-baseline-transaction"
_DOCUMENT_ID = "tool-baseline-document"
_PDF_CONTENT = b"%PDF-1.7 tool baseline test content"

_StructuredResponse = TypeVar("_StructuredResponse", bound=BaseModel)


class FakeExtractor:
    """Returns two fixed pages instead of running real Tesseract OCR."""

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
        pages = [
            ExtractedPage(
                page_number=1,
                text="Turnover for the year was 100,000.",
                character_count=35,
            ),
            ExtractedPage(
                page_number=2,
                text="The directors recommend no dividend.",
                character_count=37,
            ),
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


class FakeToolChatClient:
    """Plays back a scripted sequence of tool-calling turns, then a fixed final `Finding`.

    Each call to `complete_with_tools_and_usage` pops the next scripted
    turn; once the script is exhausted it returns a turn with no tool
    calls, ending the loop the same way a real model choosing to stop
    would. `complete_structured_with_usage` always returns `finding`,
    matching every other `Fake*ChatClient` in this project's test suite.
    """

    def __init__(
        self,
        *,
        turns: Sequence[ToolCallTurn],
        finding: Finding,
        usage: ChatUsage | None = None,
    ) -> None:
        self._turns = list(turns)
        self._finding = finding
        self._usage = usage
        self.tool_call_rounds = 0
        self.messages_at_final_call: Sequence[ChatMessage] | None = None

    async def complete_with_tools_and_usage(
        self, messages: Sequence[ChatMessage], tools: object
    ) -> tuple[ToolCallTurn, ChatUsage | None]:
        self.tool_call_rounds += 1
        if self._turns:
            return self._turns.pop(0), self._usage
        return ToolCallTurn(content="Ready to answer.", tool_calls=()), self._usage

    async def complete_structured_with_usage(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[_StructuredResponse],
    ) -> tuple[_StructuredResponse, ChatUsage | None]:
        self.messages_at_final_call = messages
        return cast(_StructuredResponse, self._finding), self._usage


class AlwaysCallingToolChatClient:
    """Never stops requesting a tool call - used to verify the loop's round budget."""

    def __init__(self, *, finding: Finding) -> None:
        self._finding = finding
        self.tool_call_rounds = 0

    async def complete_with_tools_and_usage(
        self, messages: Sequence[ChatMessage], tools: object
    ) -> tuple[ToolCallTurn, ChatUsage | None]:
        self.tool_call_rounds += 1
        call = ToolCall(
            id=f"call-{self.tool_call_rounds}", name="get_company_profile", arguments={}
        )
        return ToolCallTurn(content=None, tool_calls=(call,)), None

    async def complete_structured_with_usage(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[_StructuredResponse],
    ) -> tuple[_StructuredResponse, ChatUsage | None]:
        return cast(_StructuredResponse, self._finding), None


_Handler = Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]


def _companies_house_handler() -> _Handler:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == f"/company/{TEST_COMPANY_NUMBER}":
            return httpx2.Response(
                200,
                json={
                    "company_name": "TOOL BASELINE TEST LIMITED",
                    "company_number": TEST_COMPANY_NUMBER,
                    "type": "ltd",
                    "company_status": "active",
                    "date_of_creation": "2020-01-01",
                    "sic_codes": ["62012"],
                },
                request=request,
            )
        if request.url.path == f"/company/{TEST_COMPANY_NUMBER}/filing-history":
            return httpx2.Response(
                200,
                json={
                    "items": [
                        {
                            "transaction_id": _TRANSACTION_ID,
                            "category": "accounts",
                            "date": "2026-01-01",
                            "description": "accounts-with-accounts-type-full",
                            "type": "AA",
                            "links": {
                                "document_metadata": (
                                    "https://document-api.company-information"
                                    f".service.gov.uk/document/{_DOCUMENT_ID}"
                                )
                            },
                        }
                    ],
                    "items_per_page": 100,
                    "start_index": 0,
                    "total_count": 1,
                },
                request=request,
            )
        return httpx2.Response(404, json={"error": "not found"}, request=request)

    return handler


def _document_handler() -> _Handler:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/content"):
            return httpx2.Response(
                200,
                headers={"Content-Type": "application/pdf"},
                content=_PDF_CONTENT,
                request=request,
            )
        return httpx2.Response(
            200,
            json={
                "created_at": "2026-01-01T00:00:00Z",
                "etag": "tool-baseline-etag",
                "pages": 2,
                "resources": {"application/pdf": {"content_length": len(_PDF_CONTENT)}},
            },
            request=request,
        )

    return handler


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
                    delete(DocumentExtraction).where(
                        DocumentExtraction.filing_document_id.in_(
                            select(FilingDocument.id).where(
                                FilingDocument.filing_id.in_(
                                    select(Filing.id).where(
                                        Filing.company_number == company_number
                                    )
                                )
                            )
                        )
                    )
                )
                await cleanup_session.execute(
                    delete(Filing).where(Filing.company_number == company_number)
                )
                await cleanup_session.execute(
                    delete(Company).where(Company.company_number == company_number)
                )
            await cleanup_session.commit()
        await engine.dispose()


@pytest.fixture
def companies_house_client() -> CompaniesHouseClient:
    return CompaniesHouseClient(
        api_key="test-api-key",
        base_url="https://company.example.test",
        transport=httpx2.MockTransport(_companies_house_handler()),
    )


@pytest.fixture
def document_client() -> CompaniesHouseDocumentClient:
    return CompaniesHouseDocumentClient(
        api_key="test-api-key",
        base_url="https://document.example.test",
        transport=httpx2.MockTransport(_document_handler()),
    )


@pytest.fixture
def artifact_store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


@pytest.fixture
def extractor() -> FakeExtractor:
    return FakeExtractor()


async def _run(
    session: AsyncSession,
    chat_client: object,
    companies_house_client: CompaniesHouseClient,
    document_client: CompaniesHouseDocumentClient,
    artifact_store: LocalArtifactStore,
    extractor: FakeExtractor,
    question: str = "What was the turnover for the year?",
) -> ToolBaselineAnswer:
    return await answer_with_tools(
        session,
        chat_client,  # type: ignore[arg-type]
        companies_house_client,
        document_client,
        artifact_store,
        extractor,
        question,
        TEST_COMPANY_NUMBER,
    )


@pytest.mark.asyncio
async def test_answer_with_tools_completes_a_full_loop_and_returns_a_grounded_finding(
    session: AsyncSession,
    companies_house_client: CompaniesHouseClient,
    document_client: CompaniesHouseDocumentClient,
    artifact_store: LocalArtifactStore,
    extractor: FakeExtractor,
) -> None:
    finding = Finding(
        claim="Turnover for the year was 100,000.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=-1,  # placeholder, replaced below
                page_number=1,
                supporting_text="Turnover for the year was 100,000.",
            )
        ],
    )

    turns = [
        ToolCallTurn(
            content=None,
            tool_calls=(
                ToolCall(id="call-1", name="get_company_profile", arguments={}),
            ),
        ),
        ToolCallTurn(
            content=None,
            tool_calls=(
                ToolCall(id="call-2", name="get_filing_history", arguments={}),
            ),
        ),
        ToolCallTurn(
            content=None,
            tool_calls=(
                ToolCall(
                    id="call-3",
                    name="list_filing_document_pages",
                    arguments={"transaction_id": _TRANSACTION_ID},
                ),
            ),
        ),
    ]

    # `document_extraction_id` is only known after `list_filing_document_pages`
    # runs, so the final `get_filing_document_page_text` call and citation
    # are patched in via a first, exploratory run's result.
    probe_client = FakeToolChatClient(turns=list(turns), finding=finding)
    with pytest.raises(ToolBaselineAgentError):
        await _run(
            session,
            probe_client,
            companies_house_client,
            document_client,
            artifact_store,
            extractor,
        )

    extraction = await session.scalar(
        select(DocumentExtraction)
        .join(
            FilingDocument, DocumentExtraction.filing_document_id == FilingDocument.id
        )
        .join(Filing, FilingDocument.filing_id == Filing.id)
        .where(Filing.company_number == TEST_COMPANY_NUMBER)
    )
    assert extraction is not None
    document_extraction_id = extraction.id

    grounded_finding = Finding(
        claim="Turnover for the year was 100,000.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=document_extraction_id,
                page_number=1,
                supporting_text="Turnover for the year was 100,000.",
            )
        ],
    )
    full_turns = [
        *turns,
        ToolCallTurn(
            content=None,
            tool_calls=(
                ToolCall(
                    id="call-4",
                    name="get_filing_document_page_text",
                    arguments={
                        "document_extraction_id": document_extraction_id,
                        "page_number": 1,
                    },
                ),
            ),
        ),
    ]
    usage = ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chat_client = FakeToolChatClient(
        turns=full_turns, finding=grounded_finding, usage=usage
    )

    answer = await _run(
        session,
        chat_client,
        companies_house_client,
        document_client,
        artifact_store,
        extractor,
    )

    assert answer.finding == grounded_finding
    assert answer.tool_calls_made == 4
    assert answer.usage is not None
    # 4 tool-call rounds + 1 stopping round + 1 final structured call, each
    # reporting the fixed 15-token usage.
    assert answer.usage.total_tokens == 15 * 6
    assert chat_client.messages_at_final_call is not None


@pytest.mark.asyncio
async def test_answer_with_tools_rejects_a_citation_to_an_unread_page(
    session: AsyncSession,
    companies_house_client: CompaniesHouseClient,
    document_client: CompaniesHouseDocumentClient,
    artifact_store: LocalArtifactStore,
    extractor: FakeExtractor,
) -> None:
    finding = Finding(
        claim="Turnover for the year was 100,000.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=999999,
                page_number=1,
                supporting_text="Turnover for the year was 100,000.",
            )
        ],
    )
    chat_client = FakeToolChatClient(turns=[], finding=finding)

    with pytest.raises(ToolBaselineAgentError):
        await _run(
            session,
            chat_client,
            companies_house_client,
            document_client,
            artifact_store,
            extractor,
        )


@pytest.mark.asyncio
async def test_answer_with_tools_stops_after_the_tool_call_round_budget(
    session: AsyncSession,
    companies_house_client: CompaniesHouseClient,
    document_client: CompaniesHouseDocumentClient,
    artifact_store: LocalArtifactStore,
    extractor: FakeExtractor,
) -> None:
    finding = Finding(
        claim="Insufficient evidence gathered.",
        claim_type="fact",
        evidence_sufficient=False,
        citations=[],
    )
    chat_client = AlwaysCallingToolChatClient(finding=finding)

    answer = await _run(
        session,
        chat_client,
        companies_house_client,
        document_client,
        artifact_store,
        extractor,
    )

    assert answer.finding == finding
    assert chat_client.tool_call_rounds == 8


@pytest.mark.asyncio
async def test_get_filing_document_page_text_reports_an_error_for_a_missing_page(
    session: AsyncSession,
    companies_house_client: CompaniesHouseClient,
    document_client: CompaniesHouseDocumentClient,
    artifact_store: LocalArtifactStore,
    extractor: FakeExtractor,
) -> None:
    finding = Finding(
        claim="Could not find that page.",
        claim_type="fact",
        evidence_sufficient=False,
        citations=[],
    )
    turns = [
        ToolCallTurn(
            content=None,
            tool_calls=(
                ToolCall(
                    id="call-1",
                    name="get_filing_document_page_text",
                    arguments={"document_extraction_id": 999999, "page_number": 1},
                ),
            ),
        ),
    ]
    chat_client = FakeToolChatClient(turns=turns, finding=finding)

    answer = await _run(
        session,
        chat_client,
        companies_house_client,
        document_client,
        artifact_store,
        extractor,
    )

    assert answer.finding == finding
    assert answer.tool_calls_made == 1
    assert chat_client.messages_at_final_call is not None
    tool_result_messages = [
        message
        for message in chat_client.messages_at_final_call
        if message.role == "tool"
    ]
    assert len(tool_result_messages) == 1
    assert "error" in tool_result_messages[0].content


async def _create_other_companys_page(session: AsyncSession) -> int:
    """Persist a real page under a company distinct from TEST_COMPANY_NUMBER.

    Returns its document_extraction_id -- a real, valid id that just
    belongs to the wrong company, the case _get_filing_document_page_text
    must reject rather than silently returning.
    """
    now = datetime.now(UTC)
    session.add(
        Company(
            company_number=OTHER_TEST_COMPANY_NUMBER,
            company_name="OTHER TOOL BASELINE TEST LIMITED",
            type="ltd",
            sic_codes=[],
            raw_profile={},
            retrieved_at=now,
        )
    )
    await session.flush()
    filing = Filing(
        company_number=OTHER_TEST_COMPANY_NUMBER,
        transaction_id="other-company-transaction",
        category="accounts",
        type="AA",
        description="accounts",
        date=now.date(),
        raw_filing={},
        retrieved_at=now,
    )
    session.add(filing)
    await session.flush()
    document = FilingDocument(
        filing_id=filing.id,
        source_document_id="other-company-document",
        media_type="application/pdf",
        content_length=1234,
        sha256="b" * 64,
        storage_key="sha256/other-company-test.pdf",
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
    text = "Confidential turnover figures for the other company."
    session.add(
        DocumentPage(
            document_extraction_id=extraction.id,
            page_number=1,
            text=text,
            character_count=len(text),
        )
    )
    await session.commit()
    return extraction.id


@pytest.mark.asyncio
async def test_get_filing_document_page_text_rejects_a_page_belonging_to_another_company(
    session: AsyncSession,
    companies_house_client: CompaniesHouseClient,
    document_client: CompaniesHouseDocumentClient,
    artifact_store: LocalArtifactStore,
    extractor: FakeExtractor,
) -> None:
    """Regression test for a real found bug: document_extraction_id is a
    globally unique id shared across every company's filings in the same
    table, and the model supplies it directly as a tool-call argument
    rather than it being resolved from a prior call already scoped to this
    run's own company. Without re-checking company ownership here, a
    hallucinated or misremembered id belonging to a different company would
    silently return that company's real page text, get added to
    pages_read, and pass _validate_tool_citations as if it were genuine
    evidence for the company under investigation -- exactly the same class
    of cross-company leak previously found and fixed in
    document_extraction_ids_for_fiscal_year (see AGENTS.md).
    """
    other_companys_extraction_id = await _create_other_companys_page(session)
    finding = Finding(
        claim="Could not find that page.",
        claim_type="fact",
        evidence_sufficient=False,
        citations=[],
    )
    turns = [
        ToolCallTurn(
            content=None,
            tool_calls=(
                ToolCall(
                    id="call-1",
                    name="get_filing_document_page_text",
                    arguments={
                        "document_extraction_id": other_companys_extraction_id,
                        "page_number": 1,
                    },
                ),
            ),
        ),
    ]
    chat_client = FakeToolChatClient(turns=turns, finding=finding)

    answer = await _run(
        session,
        chat_client,
        companies_house_client,
        document_client,
        artifact_store,
        extractor,
    )

    assert answer.finding == finding
    assert chat_client.messages_at_final_call is not None
    tool_result_messages = [
        message
        for message in chat_client.messages_at_final_call
        if message.role == "tool"
    ]
    assert len(tool_result_messages) == 1
    assert "error" in tool_result_messages[0].content
    assert "Confidential turnover figures" not in tool_result_messages[0].content

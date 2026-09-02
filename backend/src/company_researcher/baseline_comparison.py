import time
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.artifact_store import ArtifactStore
from company_researcher.baseline_agent import answer_without_retrieval
from company_researcher.companies_house import (
    CompaniesHouseClient,
    CompaniesHouseDocumentClient,
)
from company_researcher.db.models import DocumentPage
from company_researcher.investigation_agent import (
    Citation,
    Finding,
    InvestigationAgentError,
    investigate_with_usage,
)
from company_researcher.llm_client import ChatUsage, FullChatProvider
from company_researcher.pdf_extraction import PdfExtractor
from company_researcher.retrieval_evaluation import (
    EvaluationDataset,
    EvaluationQuestion,
)
from company_researcher.tool_baseline_agent import (
    ToolBaselineAgentError,
    answer_with_tools,
)


@dataclass(frozen=True)
class CitationRealism:
    """Whether one citation's (document_extraction_id, page_number) exists in the persisted corpus at all.

    Deterministic, not an LLM judge: a citation either points at a page
    that exists in the database or it does not. This deliberately checks
    existence only, not whether the page was actually retrieved for this
    run (the check `_validate_citations` in `investigation_agent.py`
    performs for the specialized agent) - the baseline path has no
    retrieval step, so the only fair question to ask of it is whether an
    attempted citation refers to a real page anywhere in the corpus.
    """

    citation: Citation
    exists: bool


@dataclass(frozen=True)
class QuestionComparison:
    """One question answered by the no-retrieval baseline, the tool-using baseline, and the specialized agent."""

    question_id: str
    question_text: str
    baseline_finding: Finding
    baseline_usage: ChatUsage | None
    baseline_latency_seconds: float
    baseline_citation_realism: tuple[CitationRealism, ...]
    tool_baseline_finding: Finding | None
    tool_baseline_usage: ChatUsage | None
    tool_baseline_error: str | None
    tool_baseline_latency_seconds: float
    tool_baseline_tool_calls_made: int
    specialized_finding: Finding | None
    specialized_usage: ChatUsage | None
    specialized_error: str | None
    specialized_latency_seconds: float


async def _citation_realism(
    session: AsyncSession, citations: Sequence[Citation]
) -> tuple[CitationRealism, ...]:
    """Check each citation's (document_extraction_id, page_number) against real, persisted pages."""
    if not citations:
        return ()

    keys = [
        (citation.document_extraction_id, citation.page_number)
        for citation in citations
    ]
    statement = select(
        DocumentPage.document_extraction_id, DocumentPage.page_number
    ).where(
        tuple_(DocumentPage.document_extraction_id, DocumentPage.page_number).in_(keys)
    )
    result = await session.execute(statement)
    existing = {(row.document_extraction_id, row.page_number) for row in result}

    return tuple(
        CitationRealism(
            citation=citation,
            exists=(citation.document_extraction_id, citation.page_number) in existing,
        )
        for citation in citations
    )


async def compare_question(
    session: AsyncSession,
    chat_client: FullChatProvider,
    companies_house_client: CompaniesHouseClient,
    document_client: CompaniesHouseDocumentClient,
    artifact_store: ArtifactStore,
    extractor: PdfExtractor,
    question: EvaluationQuestion,
    company_number: str,
    company_name: str,
) -> QuestionComparison:
    """Answer one question with the no-retrieval baseline, the tool-using baseline, and the specialized agent.

    A specialized-agent `InvestigationAgentError`, or a tool-baseline
    `ToolBaselineAgentError`, is caught, not raised: an agent refusing to
    serve a fabricated or ungrounded citation is itself part of what this
    comparison measures (see README.md's "A known limitation" section for
    a real case of exactly this for the specialized agent), not a failure
    of the comparison run.
    """
    baseline_start = time.monotonic()
    baseline_answer = await answer_without_retrieval(
        chat_client, question.text, company_name
    )
    baseline_latency = time.monotonic() - baseline_start
    baseline_realism = await _citation_realism(
        session, baseline_answer.finding.citations
    )

    tool_baseline_finding: Finding | None = None
    tool_baseline_usage: ChatUsage | None = None
    tool_baseline_error: str | None = None
    tool_baseline_tool_calls_made = 0
    tool_baseline_start = time.monotonic()
    try:
        tool_answer = await answer_with_tools(
            session,
            chat_client,
            companies_house_client,
            document_client,
            artifact_store,
            extractor,
            question.text,
            company_number,
        )
        tool_baseline_finding = tool_answer.finding
        tool_baseline_usage = tool_answer.usage
        tool_baseline_tool_calls_made = tool_answer.tool_calls_made
    except ToolBaselineAgentError as error:
        tool_baseline_error = str(error)
    tool_baseline_latency = time.monotonic() - tool_baseline_start

    specialized_finding: Finding | None = None
    specialized_usage: ChatUsage | None = None
    specialized_error: str | None = None
    specialized_start = time.monotonic()
    try:
        specialized_finding, specialized_usage = await investigate_with_usage(
            session, chat_client, question.text, company_number
        )
    except InvestigationAgentError as error:
        specialized_error = str(error)
    specialized_latency = time.monotonic() - specialized_start

    return QuestionComparison(
        question_id=question.id,
        question_text=question.text,
        baseline_finding=baseline_answer.finding,
        baseline_usage=baseline_answer.usage,
        baseline_latency_seconds=baseline_latency,
        baseline_citation_realism=baseline_realism,
        tool_baseline_finding=tool_baseline_finding,
        tool_baseline_usage=tool_baseline_usage,
        tool_baseline_error=tool_baseline_error,
        tool_baseline_latency_seconds=tool_baseline_latency,
        tool_baseline_tool_calls_made=tool_baseline_tool_calls_made,
        specialized_finding=specialized_finding,
        specialized_usage=specialized_usage,
        specialized_error=specialized_error,
        specialized_latency_seconds=specialized_latency,
    )


async def run_comparison(
    session: AsyncSession,
    chat_client: FullChatProvider,
    companies_house_client: CompaniesHouseClient,
    document_client: CompaniesHouseDocumentClient,
    artifact_store: ArtifactStore,
    extractor: PdfExtractor,
    dataset: EvaluationDataset,
) -> list[QuestionComparison]:
    """Run the baseline-vs-tool-baseline-vs-specialized comparison over every question in a dataset."""
    return [
        await compare_question(
            session,
            chat_client,
            companies_house_client,
            document_client,
            artifact_store,
            extractor,
            question,
            dataset.company_number,
            dataset.company_name,
        )
        for question in dataset.questions
    ]

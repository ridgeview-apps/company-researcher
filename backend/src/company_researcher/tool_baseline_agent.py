import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.artifact_store import ArtifactStore
from company_researcher.companies_house import (
    CompaniesHouseClient,
    CompaniesHouseDocumentClient,
    normalize_company_number,
)
from company_researcher.db.models import (
    Company,
    DocumentExtraction,
    DocumentPage,
    Filing,
    FilingDocument,
)
from company_researcher.document_ingestion import ingest_filing_document
from company_researcher.extraction_persistence import extract_filing_document
from company_researcher.ingestion import ingest_company
from company_researcher.investigation_agent import Citation, Finding
from company_researcher.llm_client import (
    ChatMessage,
    ChatUsage,
    ToolAwareChatProvider,
    ToolCall,
    ToolDefinition,
)
from company_researcher.pdf_extraction import PdfExtractor

_MAX_TOOL_CALL_ROUNDS = 8

_TOOL_BASELINE_SYSTEM_PROMPT = (
    "You are answering a question about a specific UK company. You have "
    "real tool access to that company's own Companies House record: its "
    "profile, its filing history, and the OCR'd text of its filing "
    "documents. Decide for yourself which filings and pages are relevant "
    "and read them with the tools before answering - do not guess or rely "
    "on general knowledge when a tool could confirm the fact. When you "
    "have enough evidence, respond with your final answer instead of "
    "calling another tool. Every citation must give the exact "
    "document_extraction_id and page_number of a page you actually read "
    "with get_filing_document_page_text, with an exact quote copied from "
    "that page as supporting_text - never invent one, and never cite a "
    "page you have not read with that tool. Set evidence_sufficient to "
    "false if the filings do not establish a confident answer, rather "
    "than guessing. Classify your answer with claim_type, either 'fact' "
    "(states only what the filings establish) or 'interpretation' (adds a "
    "judgement beyond the filings' own statements)."
)

_GET_COMPANY_PROFILE_TOOL = ToolDefinition(
    name="get_company_profile",
    description=(
        "Get the company's current structured profile: name, type, status, "
        "incorporation date, and SIC codes."
    ),
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)

_GET_FILING_HISTORY_TOOL = ToolDefinition(
    name="get_filing_history",
    description=(
        "List the company's filing history: transaction_id, category, "
        "type, description, and date for each filing, and whether it has "
        "a downloadable document. Optionally filter by category (e.g. "
        "'accounts', 'mortgage', 'officers')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional Companies House filing category to filter by.",
            }
        },
        "additionalProperties": False,
    },
)

_LIST_FILING_DOCUMENT_PAGES_TOOL = ToolDefinition(
    name="list_filing_document_pages",
    description=(
        "Download and OCR one filing's source document (reusing a "
        "previous extraction if this filing was already read), returning "
        "its document_extraction_id, page count, and a short snippet of "
        "each page so you can decide which pages are worth reading in "
        "full with get_filing_document_page_text."
    ),
    parameters={
        "type": "object",
        "properties": {
            "transaction_id": {
                "type": "string",
                "description": "The filing's transaction_id, from get_filing_history.",
            }
        },
        "required": ["transaction_id"],
        "additionalProperties": False,
    },
)

_GET_FILING_DOCUMENT_PAGE_TEXT_TOOL = ToolDefinition(
    name="get_filing_document_page_text",
    description="Get the full OCR text of one specific page of a filing document.",
    parameters={
        "type": "object",
        "properties": {
            "document_extraction_id": {"type": "integer"},
            "page_number": {"type": "integer"},
        },
        "required": ["document_extraction_id", "page_number"],
        "additionalProperties": False,
    },
)

_TOOLS: tuple[ToolDefinition, ...] = (
    _GET_COMPANY_PROFILE_TOOL,
    _GET_FILING_HISTORY_TOOL,
    _LIST_FILING_DOCUMENT_PAGES_TOOL,
    _GET_FILING_DOCUMENT_PAGE_TEXT_TOOL,
)


class ToolBaselineAgentError(Exception):
    """Raised when the tool-using baseline cites a page it never actually read."""


class _ToolExecutionError(Exception):
    """Raised for a tool-call failure that should be reported back to the model, not crash the run."""


@dataclass(frozen=True)
class ToolBaselineAnswer:
    """One question answered by the tool-using baseline, for comparison against the other baselines."""

    finding: Finding
    usage: ChatUsage | None
    tool_calls_made: int


@dataclass
class _ToolBaselineContext:
    """Mutable state threaded through one tool-baseline run's tool executions."""

    session: AsyncSession
    document_client: CompaniesHouseDocumentClient
    artifact_store: ArtifactStore
    extractor: PdfExtractor
    company_number: str
    pages_read: set[tuple[int, int]] = field(default_factory=set)


async def answer_with_tools(
    session: AsyncSession,
    chat_client: ToolAwareChatProvider,
    companies_house_client: CompaniesHouseClient,
    document_client: CompaniesHouseDocumentClient,
    artifact_store: ArtifactStore,
    extractor: PdfExtractor,
    question: str,
    company_number: str,
) -> ToolBaselineAnswer:
    """Answer one question using a bounded tool-calling loop over real Companies House data.

    This is the project brief's "General LLM + web, instructed to use
    Companies House" baseline, scoped to Companies House itself rather than
    open web search (no new provider/secret, directly comparable
    citations, reproducible). Unlike `answer_without_retrieval` (no tools
    at all) or `investigate()` (engineered query generation and restricted
    lexical retrieval over an already-OCR'd corpus), this baseline gives
    the model real tools - the company's profile, its filing history, and
    on-demand OCR of any filing document - and lets it decide for itself
    what to fetch and read. That isolates whether this project's engineered
    retrieval/verification machinery earns its keep against a general
    tool-using agent working from the same underlying data, not different
    data.

    The tools reuse this project's existing ingestion/OCR pipeline
    unchanged (`ingest_company`, `ingest_filing_document`,
    `extract_filing_document`), matching the project's principle that the
    domain-specific data layer stays separate from - and here, shared
    with - the reusable AI architecture being compared.
    """
    normalized_company_number = normalize_company_number(company_number)
    await ingest_company(session, companies_house_client, normalized_company_number)

    context = _ToolBaselineContext(
        session=session,
        document_client=document_client,
        artifact_store=artifact_store,
        extractor=extractor,
        company_number=normalized_company_number,
    )

    messages: list[ChatMessage] = [
        ChatMessage(role="system", content=_TOOL_BASELINE_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=f"Company number: {normalized_company_number}\n\nQuestion: {question}",
        ),
    ]
    usage_records: list[ChatUsage] = []
    tool_calls_made = 0

    for _ in range(_MAX_TOOL_CALL_ROUNDS):
        turn, usage = await chat_client.complete_with_tools_and_usage(messages, _TOOLS)
        if usage is not None:
            usage_records.append(usage)
        if not turn.tool_calls:
            break

        messages.append(
            ChatMessage(
                role="assistant", content=turn.content or "", tool_calls=turn.tool_calls
            )
        )
        for call in turn.tool_calls:
            tool_calls_made += 1
            result_text = await _execute_tool_call(context, call)
            messages.append(
                ChatMessage(role="tool", content=result_text, tool_call_id=call.id)
            )
    else:
        messages.append(
            ChatMessage(
                role="user",
                content=(
                    "You have reached the tool-call limit. Answer now with "
                    "the evidence you have already gathered."
                ),
            )
        )

    finding, structured_usage = await chat_client.complete_structured_with_usage(
        messages, Finding
    )
    if structured_usage is not None:
        usage_records.append(structured_usage)

    _validate_tool_citations(finding.citations, context.pages_read)

    return ToolBaselineAnswer(
        finding=finding,
        usage=_sum_usage(usage_records),
        tool_calls_made=tool_calls_made,
    )


def _sum_usage(records: Sequence[ChatUsage]) -> ChatUsage | None:
    """Sum token usage across every LLM call in one run. See `investigation_agent._sum_usage`."""
    if not records:
        return None
    return ChatUsage(
        prompt_tokens=sum(record.prompt_tokens for record in records),
        completion_tokens=sum(record.completion_tokens for record in records),
        total_tokens=sum(record.total_tokens for record in records),
    )


def _validate_tool_citations(
    citations: Sequence[Citation], pages_read: set[tuple[int, int]]
) -> None:
    """Reject any citation to a page this run's tools never actually returned.

    The same discipline `investigation_agent._validate_citations` applies
    to the specialized agent's citations against its retrieved pages, so
    the two baselines stay comparable on citation groundedness, not just
    on citation existence somewhere in the corpus (the weaker check
    `baseline_comparison._citation_realism` applies to the no-tool
    baseline, which has no retrieved/read-page set to check against).
    """
    for citation in citations:
        key = (citation.document_extraction_id, citation.page_number)
        if key not in pages_read:
            raise ToolBaselineAgentError(
                f"Finding cited document_extraction_id={key[0]} "
                f"page_number={key[1]}, which this tool-baseline run never "
                "actually read"
            )


async def _execute_tool_call(context: _ToolBaselineContext, call: ToolCall) -> str:
    """Execute one tool call and return its JSON-encoded result for the model."""
    try:
        if call.name == "get_company_profile":
            result = await _get_company_profile(context)
        elif call.name == "get_filing_history":
            category = call.arguments.get("category")
            result = await _get_filing_history(
                context, category if isinstance(category, str) else None
            )
        elif call.name == "list_filing_document_pages":
            result = await _list_filing_document_pages(
                context, str(call.arguments["transaction_id"])
            )
        elif call.name == "get_filing_document_page_text":
            result = await _get_filing_document_page_text(
                context,
                _coerce_int(
                    call.arguments["document_extraction_id"],
                    field="document_extraction_id",
                ),
                _coerce_int(call.arguments["page_number"], field="page_number"),
            )
        else:
            result = {"error": f"Unknown tool: {call.name}"}
    except _ToolExecutionError as error:
        result = {"error": str(error)}
    except KeyError as error:
        result = {"error": f"Missing required argument: {error}"}
    return json.dumps(result, ensure_ascii=False)


def _coerce_int(value: object, *, field: str) -> int:
    """Coerce one JSON-decoded tool argument to an int, tolerating the model sending a numeric string."""
    if isinstance(value, bool):
        raise _ToolExecutionError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    raise _ToolExecutionError(f"{field} must be an integer")


async def _get_company_profile(context: _ToolBaselineContext) -> dict[str, object]:
    company = await context.session.get(Company, context.company_number)
    if company is None:
        raise _ToolExecutionError("Company is not persisted")
    return {
        "company_number": company.company_number,
        "company_name": company.company_name,
        "type": company.type,
        "company_status": company.company_status,
        "date_of_creation": (
            company.date_of_creation.isoformat() if company.date_of_creation else None
        ),
        "date_of_cessation": (
            company.date_of_cessation.isoformat() if company.date_of_cessation else None
        ),
        "sic_codes": company.sic_codes,
    }


async def _get_filing_history(
    context: _ToolBaselineContext, category: str | None
) -> dict[str, object]:
    statement = select(Filing).where(Filing.company_number == context.company_number)
    if category:
        statement = statement.where(Filing.category == category)
    statement = statement.order_by(Filing.date.desc())
    filings = list(await context.session.scalars(statement))
    return {
        "filings": [
            {
                "transaction_id": filing.transaction_id,
                "category": filing.category,
                "type": filing.type,
                "description": filing.description,
                "date": filing.date.isoformat(),
                "has_document": filing.source_document_id is not None,
            }
            for filing in filings
        ]
    }


async def _list_filing_document_pages(
    context: _ToolBaselineContext, transaction_id: str
) -> dict[str, object]:
    filing = await context.session.scalar(
        select(Filing).where(
            Filing.company_number == context.company_number,
            Filing.transaction_id == transaction_id,
        )
    )
    if filing is None:
        raise _ToolExecutionError(f"No filing with transaction_id={transaction_id!r}")
    if filing.source_document_id is None:
        raise _ToolExecutionError("This filing has no downloadable document")

    ingestion_result = await ingest_filing_document(
        context.session, context.document_client, context.artifact_store, filing
    )
    filing_document = await context.session.get(
        FilingDocument, ingestion_result.filing_document_id
    )
    if filing_document is None:
        raise _ToolExecutionError("Filing document was not persisted")

    extraction_result = await extract_filing_document(
        context.session, context.artifact_store, context.extractor, filing_document
    )
    pages = list(
        await context.session.scalars(
            select(DocumentPage)
            .where(
                DocumentPage.document_extraction_id
                == extraction_result.document_extraction_id
            )
            .order_by(DocumentPage.page_number)
        )
    )
    return {
        "document_extraction_id": extraction_result.document_extraction_id,
        "page_count": extraction_result.page_count,
        "pages": [
            {
                "page_number": page.page_number,
                "snippet": page.text[:150].replace("\n", " "),
            }
            for page in pages
        ],
    }


async def _get_filing_document_page_text(
    context: _ToolBaselineContext, document_extraction_id: int, page_number: int
) -> dict[str, object]:
    """Return one page's OCR'd text, scoped to this run's own company.

    document_extraction_id is a globally unique id shared across every
    company's filings in the same table, and this argument is supplied
    directly by the model rather than resolved from a prior scoped tool
    call - it must be re-checked against context.company_number here,
    the same discipline the other three tools already apply, or a
    hallucinated/misremembered id belonging to a different company would
    silently return that company's real page text, get added to
    pages_read, and pass _validate_tool_citations as if it were genuine
    evidence for the company under investigation.
    """
    page = await context.session.scalar(
        select(DocumentPage)
        .join(
            DocumentExtraction,
            DocumentExtraction.id == DocumentPage.document_extraction_id,
        )
        .join(
            FilingDocument, FilingDocument.id == DocumentExtraction.filing_document_id
        )
        .join(Filing, Filing.id == FilingDocument.filing_id)
        .where(
            DocumentPage.document_extraction_id == document_extraction_id,
            DocumentPage.page_number == page_number,
            Filing.company_number == context.company_number,
        )
    )
    if page is None:
        raise _ToolExecutionError(
            f"No page {page_number} for "
            f"document_extraction_id={document_extraction_id}. Call "
            "list_filing_document_pages first to see valid page numbers."
        )
    context.pages_read.add((document_extraction_id, page_number))
    return {"text": page.text}

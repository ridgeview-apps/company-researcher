import re
from collections.abc import Sequence
from typing import TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import DocumentPage
from company_researcher.fiscal_year_extraction import extract_fiscal_years
from company_researcher.fiscal_year_lookup import (
    document_extraction_ids_for_fiscal_year,
)
from company_researcher.lexical_search import PageMatch, search_pages
from company_researcher.llm_client import ChatMessage, ChatProvider

DEFAULT_SEARCH_DEPTH = 50
DEFAULT_CONTEXT_PAGES = 5

_QUERY_SYSTEM_PROMPT = (
    "You are helping search a corpus of scanned UK statutory accounts "
    "filings using PostgreSQL full-text search. Produce a short search "
    "query of a few specific, discriminative keywords or phrases likely to "
    "appear verbatim on the single most relevant page - not a paraphrase of "
    "the question, and not generic words that recur on most pages of an "
    "accounts filing (such as 'accounts', 'company', 'financial'). Filings "
    "write fiscal years as plain numbers (e.g. '2023'), never with an 'FY' "
    "prefix, so use plain years too. Respond with only the query text: no "
    "punctuation, quotes, or explanation."
)

_FINDING_SYSTEM_PROMPT = (
    "You are an evidence-driven investigation assistant. Answer the "
    "question using ONLY the evidence pages provided below. Every citation "
    "must reference one of the listed pages, using its exact "
    "document_extraction_id and page_number - never cite a page that is not "
    "listed. If the provided pages do not contain enough information to "
    "answer confidently, set evidence_sufficient to false and say so in the "
    "claim rather than guessing or inventing an explanation. UK statutory "
    "accounts filings contain multiple distinct voices - for example the "
    "directors' own report and notes, and the independent auditor's report "
    "- which often discuss the same topic (such as going concern) on nearby "
    "pages without being interchangeable. When the question asks what a "
    "specific party stated or identified, rely only on that party's own "
    "words; do not attribute the auditor's opinion or wording to the "
    "directors, or vice versa, even where both discuss the same topic."
)

_AGGREGATE_SYSTEM_PROMPT = (
    "You are an evidence-driven investigation assistant producing a final "
    "answer that compares or explains a trend across multiple fiscal years. "
    "You will be given each fiscal year's already-grounded claim, whether "
    "its evidence was sufficient, and its available citations. Synthesize "
    "one overall claim addressing the original question across all of the "
    "years - explicitly note any year for which no evidence was found "
    "rather than omitting it silently. Every citation in your response must "
    "be copied exactly (document_extraction_id, page_number, and "
    "supporting_text) from the citations listed below for the relevant "
    "year - do not invent a new citation or alter any of its fields. If "
    "none of the per-year findings provide enough evidence to support a "
    "comparison, set evidence_sufficient to false and say so."
)


class InvestigationAgentError(Exception):
    """Raised when the agent produces a finding that violates its evidence contract."""


def _force_unambiguous_fiscal_year(query: str, question: str) -> str:
    """Append the question's fiscal year to `query` when exactly one is named.

    `generate_query`'s LLM call does not reliably include a literal year
    token in its generated query, and lexical search's OR-combined
    `ts_rank` needs that literal token to disambiguate near-identical
    boilerplate across fiscal years (see README's "Run the investigation
    agent" section for the observed failure). Only applied when the
    question names exactly one year: the evaluation dataset's hand-tuned
    queries for multi-year range questions (e.g. "FY2021 through FY2025")
    deliberately omit any year at all, so forcing one in here for those
    would diverge from that established, measured-good behaviour instead
    of fixing the single-year case that actually failed.
    """
    years = extract_fiscal_years(question)
    if len(years) != 1:
        return query
    year = years[0]
    if re.search(rf"\b{year}\b", query):
        return query
    return f"{query} {year}".strip()


def _fiscal_year_range(years: Sequence[str]) -> list[str]:
    """Expand 2+ named years into the inclusive range between the earliest and latest.

    A multi-year question names only its boundary years as literal tokens
    (e.g. "FY2021 through FY2025" yields only "2021" and "2025" from
    `extract_fiscal_years`), but the evaluation dataset's own multi-year
    questions expect evidence from *every* year in between, not just the
    endpoints (q4's answer covers FY2021, FY2022, FY2023, and FY2025
    individually). Returns an empty list for 0 or 1 named years, matching
    `_force_unambiguous_fiscal_year`'s existing single-year/no-year
    threshold, since those cases are already handled by the single-pass
    retrieval path.
    """
    if len(years) < 2:
        return []
    year_ints = sorted(int(year) for year in years)
    return [str(year) for year in range(year_ints[0], year_ints[-1] + 1)]


class RetrievedPage(BaseModel):
    """One page of OCR text retrieved as candidate evidence for a question."""

    document_extraction_id: int
    page_number: int
    text: str


class Citation(BaseModel):
    """A single piece of evidence supporting a finding's claim."""

    model_config = ConfigDict(extra="forbid")

    document_extraction_id: int
    page_number: int
    supporting_text: str


class Finding(BaseModel):
    """A structured, citation-grounded answer to one investigation question."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    evidence_sufficient: bool
    citations: list[Citation]


class YearEvidence(BaseModel):
    """One fiscal year's independently retrieved evidence and grounded sub-finding.

    Kept separate per year so each sub-finding is synthesized from only
    that year's own retrieved pages - the same discipline that fixed the
    single-question cross-fiscal-year citation leak, now applied to a
    genuinely multi-year question instead of relying on one shared,
    mixed-year context window.
    """

    fiscal_year: str
    retrieved_pages: list[RetrievedPage]
    finding: Finding


class InvestigationState(TypedDict, total=False):
    """LangGraph state threaded through the investigation graph."""

    question: str
    generated_query: str
    fiscal_year: str | None
    fiscal_year_range: list[str]
    retrieved_pages: list[RetrievedPage]
    year_evidence: list[YearEvidence]
    finding: Finding


async def _load_page_texts(
    session: AsyncSession, matches: Sequence[PageMatch]
) -> list[RetrievedPage]:
    """Fetch page text for a ranked set of lexical matches, preserving their order."""
    if not matches:
        return []

    keys = [(match.document_extraction_id, match.page_number) for match in matches]
    statement = select(
        DocumentPage.document_extraction_id, DocumentPage.page_number, DocumentPage.text
    ).where(
        tuple_(DocumentPage.document_extraction_id, DocumentPage.page_number).in_(keys)
    )
    result = await session.execute(statement)
    text_by_key = {
        (row.document_extraction_id, row.page_number): row.text for row in result
    }

    return [
        RetrievedPage(
            document_extraction_id=key[0], page_number=key[1], text=text_by_key[key]
        )
        for key in keys
        if key in text_by_key
    ]


def _validate_citations(
    finding: Finding, retrieved_pages: Sequence[RetrievedPage]
) -> None:
    """Reject a finding that cites a page outside the evidence it was actually given."""
    available = {
        (page.document_extraction_id, page.page_number) for page in retrieved_pages
    }
    for citation in finding.citations:
        key = (citation.document_extraction_id, citation.page_number)
        if key not in available:
            raise InvestigationAgentError(
                f"Finding cited document_extraction_id={key[0]} "
                f"page_number={key[1]}, which was not part of the retrieved evidence"
            )


def _format_evidence_text(pages: Sequence[RetrievedPage], *, empty_message: str) -> str:
    """Render retrieved pages as labelled evidence text for a synthesis prompt."""
    if not pages:
        return empty_message
    return "\n\n".join(
        f"[document_extraction_id={page.document_extraction_id} "
        f"page_number={page.page_number}]\n{page.text}"
        for page in pages
    )


def _format_year_findings_summary(year_evidence: Sequence[YearEvidence]) -> str:
    """Render each year's already-grounded sub-finding for the aggregation prompt.

    Passes only each sub-finding's claim, sufficiency, and citations - not
    the raw page text again - since grounding already happened once per
    year; the aggregation step is a narrative/comparison layer over facts
    already validated, not a second pass over OCR text.
    """
    return "\n\n".join(
        f"Fiscal year {evidence.fiscal_year}:\n"
        f"  claim: {evidence.finding.claim}\n"
        f"  evidence_sufficient: {evidence.finding.evidence_sufficient}\n"
        f"  citations: {[citation.model_dump() for citation in evidence.finding.citations]}"
        for evidence in year_evidence
    )


def _route_after_generate_query(state: InvestigationState) -> str:
    """Send genuinely multi-year questions down the per-year gather/aggregate path."""
    if len(state.get("fiscal_year_range", [])) >= 2:
        return "gather_year_findings"
    return "retrieve_evidence"


def _build_graph(
    session: AsyncSession,
    chat_client: ChatProvider,
    *,
    search_depth: int,
    context_pages: int,
) -> CompiledStateGraph[
    InvestigationState, None, InvestigationState, InvestigationState
]:
    """Assemble the investigation graph.

    generate_query always runs first, then branches on how many fiscal
    years the question names: 0 or 1 (the original, unchanged path) goes
    through a single retrieve_evidence -> synthesize_finding pass; 2 or
    more (a genuinely multi-year question) goes through a per-year
    gather_year_findings -> aggregate_findings pass instead, so the
    question's evidence for one fiscal year is never crowded out by
    another's in a single shared context window.
    """

    async def generate_query_node(state: InvestigationState) -> InvestigationState:
        query = await chat_client.complete(
            [
                ChatMessage(role="system", content=_QUERY_SYSTEM_PROMPT),
                ChatMessage(role="user", content=state["question"]),
            ]
        )
        forced_query = _force_unambiguous_fiscal_year(query.strip(), state["question"])
        years = extract_fiscal_years(state["question"])
        fiscal_year = years[0] if len(years) == 1 else None
        return {
            "generated_query": forced_query,
            "fiscal_year": fiscal_year,
            "fiscal_year_range": _fiscal_year_range(years),
        }

    async def retrieve_evidence_node(state: InvestigationState) -> InvestigationState:
        fiscal_year = state.get("fiscal_year")
        document_extraction_ids = None
        if fiscal_year is not None:
            document_extraction_ids = await document_extraction_ids_for_fiscal_year(
                session, fiscal_year
            )
        matches = await search_pages(
            session,
            state["generated_query"],
            limit=search_depth,
            document_extraction_ids=document_extraction_ids,
        )
        pages = await _load_page_texts(session, matches[:context_pages])
        return {"retrieved_pages": pages}

    async def synthesize_finding_node(state: InvestigationState) -> InvestigationState:
        pages = state["retrieved_pages"]
        evidence_text = _format_evidence_text(
            pages, empty_message="No evidence pages were retrieved for this question."
        )
        user_message = f"Question: {state['question']}\n\nAvailable evidence pages:\n\n{evidence_text}"
        finding = await chat_client.complete_structured(
            [
                ChatMessage(role="system", content=_FINDING_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_message),
            ],
            Finding,
        )
        _validate_citations(finding, pages)
        return {"finding": finding}

    async def gather_year_findings_node(
        state: InvestigationState,
    ) -> InvestigationState:
        question = state["question"]
        query = state["generated_query"]
        year_evidence: list[YearEvidence] = []
        for year in state["fiscal_year_range"]:
            document_extraction_ids = await document_extraction_ids_for_fiscal_year(
                session, year
            )
            matches = await search_pages(
                session,
                query,
                limit=search_depth,
                document_extraction_ids=document_extraction_ids,
            )
            pages = await _load_page_texts(session, matches[:context_pages])
            evidence_text = _format_evidence_text(
                pages,
                empty_message="No evidence pages were retrieved for this fiscal year.",
            )
            user_message = (
                f"Question: {question}\n\nFocus specifically on fiscal year {year}.\n\n"
                f"Available evidence pages:\n\n{evidence_text}"
            )
            finding = await chat_client.complete_structured(
                [
                    ChatMessage(role="system", content=_FINDING_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=user_message),
                ],
                Finding,
            )
            _validate_citations(finding, pages)
            year_evidence.append(
                YearEvidence(fiscal_year=year, retrieved_pages=pages, finding=finding)
            )
        return {"year_evidence": year_evidence}

    async def aggregate_findings_node(state: InvestigationState) -> InvestigationState:
        year_evidence = state["year_evidence"]
        summary = _format_year_findings_summary(year_evidence)
        user_message = (
            f"Question: {state['question']}\n\nPer-year findings:\n\n{summary}"
        )
        finding = await chat_client.complete_structured(
            [
                ChatMessage(role="system", content=_AGGREGATE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_message),
            ],
            Finding,
        )
        all_pages = [
            page for evidence in year_evidence for page in evidence.retrieved_pages
        ]
        _validate_citations(finding, all_pages)
        return {"finding": finding}

    graph = StateGraph(InvestigationState)
    graph.add_node("generate_query", generate_query_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("synthesize_finding", synthesize_finding_node)
    graph.add_node("gather_year_findings", gather_year_findings_node)
    graph.add_node("aggregate_findings", aggregate_findings_node)
    graph.add_edge(START, "generate_query")
    graph.add_conditional_edges(
        "generate_query",
        _route_after_generate_query,
        {
            "retrieve_evidence": "retrieve_evidence",
            "gather_year_findings": "gather_year_findings",
        },
    )
    graph.add_edge("retrieve_evidence", "synthesize_finding")
    graph.add_edge("synthesize_finding", END)
    graph.add_edge("gather_year_findings", "aggregate_findings")
    graph.add_edge("aggregate_findings", END)
    return graph.compile()


async def investigate(
    session: AsyncSession,
    chat_client: ChatProvider,
    question: str,
    *,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
    context_pages: int = DEFAULT_CONTEXT_PAGES,
) -> Finding:
    """Run the investigation graph for one natural-language question.

    Uses lexical search only: on this project's measured Gymshark
    evaluation corpus, hand-tuned lexical search outperforms both
    vector-only search and naive equal-weighted RRF hybrid (see README.md),
    so lexical is the retrieval tool this first version of the agent calls.
    Unlike the evaluation dataset's hand-tuned queries, `generated_query` is
    produced by the LLM from the question alone at run time.
    """
    graph = _build_graph(
        session, chat_client, search_depth=search_depth, context_pages=context_pages
    )
    result = await graph.ainvoke({"question": question})
    return cast(Finding, result["finding"])

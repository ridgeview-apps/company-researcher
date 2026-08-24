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


class InvestigationState(TypedDict, total=False):
    """LangGraph state threaded through the investigation graph."""

    question: str
    generated_query: str
    retrieved_pages: list[RetrievedPage]
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


def _build_graph(
    session: AsyncSession,
    chat_client: ChatProvider,
    *,
    search_depth: int,
    context_pages: int,
) -> CompiledStateGraph[
    InvestigationState, None, InvestigationState, InvestigationState
]:
    """Assemble the linear generate_query -> retrieve_evidence -> synthesize_finding graph."""

    async def generate_query_node(state: InvestigationState) -> InvestigationState:
        query = await chat_client.complete(
            [
                ChatMessage(role="system", content=_QUERY_SYSTEM_PROMPT),
                ChatMessage(role="user", content=state["question"]),
            ]
        )
        forced_query = _force_unambiguous_fiscal_year(query.strip(), state["question"])
        return {"generated_query": forced_query}

    async def retrieve_evidence_node(state: InvestigationState) -> InvestigationState:
        matches = await search_pages(
            session, state["generated_query"], limit=search_depth
        )
        pages = await _load_page_texts(session, matches[:context_pages])
        return {"retrieved_pages": pages}

    async def synthesize_finding_node(state: InvestigationState) -> InvestigationState:
        pages = state["retrieved_pages"]
        evidence_text = (
            "\n\n".join(
                f"[document_extraction_id={page.document_extraction_id} "
                f"page_number={page.page_number}]\n{page.text}"
                for page in pages
            )
            if pages
            else "No evidence pages were retrieved for this question."
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

    graph = StateGraph(InvestigationState)
    graph.add_node("generate_query", generate_query_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("synthesize_finding", synthesize_finding_node)
    graph.add_edge(START, "generate_query")
    graph.add_edge("generate_query", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "synthesize_finding")
    graph.add_edge("synthesize_finding", END)
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

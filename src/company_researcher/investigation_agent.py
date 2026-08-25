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
    "listed. Every citation's supporting_text must be an exact, contiguous "
    "quote copied verbatim from that page's text below - do not paraphrase, "
    "summarize, or splice together text from different parts of the page or "
    "from different tables. If the provided pages do not contain enough "
    "information to answer confidently, set evidence_sufficient to false "
    "and say so in the claim rather than guessing or inventing an "
    "explanation. UK statutory accounts filings contain multiple distinct "
    "voices - for example the directors' own report and notes, and the "
    "independent auditor's report - which often discuss the same topic "
    "(such as going concern) on nearby pages without being interchangeable. "
    "When the question asks what a specific party stated or identified, "
    "rely only on that party's own words; do not attribute the auditor's "
    "opinion or wording to the directors, or vice versa, even where both "
    "discuss the same topic."
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
    company_number: str
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


def _normalize_for_quote_check(text: str) -> str:
    """Strip whitespace/punctuation noise and case so a genuine quote isn't rejected for it.

    Real runs against the persisted corpus (see README.md's "Verifying
    citation quotes" section) surfaced several recurring, non-substantive
    differences between a real page and an otherwise-genuine quote of it:
    OCR renders a "." instead of "," as a thousands separator (e.g.
    "437.629" for "437,629"); OCR pairs a mismatched bracket character
    (e.g. "{Appointed 9 January 2023)" for "(Appointed 9 January 2023)");
    OCR drops a space inside a word or name (e.g. "N AMcElhinney" for "N A
    McElhinney"); and the model itself naturally joins a page's newline-
    separated list (e.g. a list of directors, one name per line) into a
    comma-separated prose sentence when quoting it, terminated with a
    period the source never had. None of these involve a different word or
    digit sequence - only whitespace and punctuation - so every run of
    whitespace is removed entirely rather than merely collapsed, commas
    and periods are stripped, curly braces are canonicalized to
    parentheses, and stray underscore "leader" characters (e.g.
    "__260.674") are stripped too. This is a deliberate trade-off: it
    makes the check slightly more permissive (in principle two genuinely
    different numbers, or two adjacent but unrelated words, could collide
    once whitespace and separators between them are removed), which is
    acceptable because this check only verifies quote *fidelity* to real
    page text - catching a wrong page or fabricated content - not the
    numeric or semantic correctness of the claim built from it, which is a
    distinct, harder problem not covered here (see the real-run FY2022
    example in the same README section).
    """
    normalized = text.replace("{", "(").replace("}", ")")
    for character in (",", ".", "_"):
        normalized = normalized.replace(character, "")
    return "".join(normalized.split()).lower()


def _find_quote_mismatches(
    finding: Finding, retrieved_pages: Sequence[RetrievedPage]
) -> list[Citation]:
    """Return citations whose supporting_text is not a verbatim excerpt of its cited page.

    Assumes `_validate_citations` has already confirmed every citation's
    page was actually retrieved - a citation whose page is missing from
    `retrieved_pages` is skipped here rather than re-reported. Catches a
    citation that points at a real, retrieved page but quotes text that
    was never actually written there (including text spliced together
    from different parts of the page) - a genuine gap the existing
    page-identity check alone cannot catch, observed on a real
    investigation run (see README.md's "Verifying citation quotes"
    section).
    """
    text_by_key = {
        (page.document_extraction_id, page.page_number): page.text
        for page in retrieved_pages
    }
    mismatches = []
    for citation in finding.citations:
        page_text = text_by_key.get(
            (citation.document_extraction_id, citation.page_number)
        )
        if page_text is None:
            continue
        quote = _normalize_for_quote_check(citation.supporting_text)
        if quote and quote not in _normalize_for_quote_check(page_text):
            mismatches.append(citation)
    return mismatches


def _format_quote_correction_request(mismatches: Sequence[Citation]) -> str:
    """Describe exactly which citation quotes failed verbatim verification, for a retry prompt."""
    lines = "\n".join(
        f"- document_extraction_id={citation.document_extraction_id} "
        f"page_number={citation.page_number}: "
        f'"{citation.supporting_text}" is not an exact, contiguous quote from that page'
        for citation in mismatches
    )
    return (
        "Your previous response's supporting_text was not an exact, "
        "contiguous quote copied verbatim from the cited page's text for "
        f"the following citation(s):\n{lines}\n\n"
        "Respond again. Keep the same claim if it is still correct, but "
        "replace each supporting_text above with an exact, contiguous "
        "excerpt copied verbatim from that citation's page - do not "
        "paraphrase or splice text from different parts of the page "
        "together."
    )


async def _synthesize_and_validate(
    chat_client: ChatProvider,
    system_prompt: str,
    user_message: str,
    retrieved_pages: Sequence[RetrievedPage],
) -> Finding:
    """Run one structured synthesis call and enforce both citation guarantees.

    Every citation must reference a page that was actually retrieved
    (`_validate_citations`, unchanged, fail-closed with no retry - an
    existence violation is a more severe error than an imprecise quote).
    Every citation's supporting_text must also be a genuine, verbatim
    excerpt of that page's real text (`_find_quote_mismatches`). A failed
    quote check retries the synthesis once with feedback naming exactly
    which quote was wrong, giving the model a chance to self-correct
    before this raises `InvestigationAgentError`.
    """
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_message),
    ]
    finding = await chat_client.complete_structured(messages, Finding)
    _validate_citations(finding, retrieved_pages)
    mismatches = _find_quote_mismatches(finding, retrieved_pages)
    if not mismatches:
        return finding

    retry_message = f"{user_message}\n\n{_format_quote_correction_request(mismatches)}"
    retried_finding = await chat_client.complete_structured(
        [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=retry_message),
        ],
        Finding,
    )
    _validate_citations(retried_finding, retrieved_pages)
    remaining_mismatches = _find_quote_mismatches(retried_finding, retrieved_pages)
    if remaining_mismatches:
        citation = remaining_mismatches[0]
        raise InvestigationAgentError(
            f"Finding cited document_extraction_id={citation.document_extraction_id} "
            f"page_number={citation.page_number} with a supporting_text quote that is "
            "not verbatim text from that page, even after a self-correction retry"
        )
    return retried_finding


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
            company_number=state["company_number"],
        )
        pages = await _load_page_texts(session, matches[:context_pages])
        return {"retrieved_pages": pages}

    async def synthesize_finding_node(state: InvestigationState) -> InvestigationState:
        pages = state["retrieved_pages"]
        evidence_text = _format_evidence_text(
            pages, empty_message="No evidence pages were retrieved for this question."
        )
        user_message = f"Question: {state['question']}\n\nAvailable evidence pages:\n\n{evidence_text}"
        finding = await _synthesize_and_validate(
            chat_client, _FINDING_SYSTEM_PROMPT, user_message, pages
        )
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
                company_number=state["company_number"],
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
            finding = await _synthesize_and_validate(
                chat_client, _FINDING_SYSTEM_PROMPT, user_message, pages
            )
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
        all_pages = [
            page for evidence in year_evidence for page in evidence.retrieved_pages
        ]
        finding = await _synthesize_and_validate(
            chat_client, _AGGREGATE_SYSTEM_PROMPT, user_message, all_pages
        )
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
    company_number: str,
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

    `company_number` is required, not optional/no-op like the fiscal-year
    restriction: unlike a fiscal year, which a question may or may not
    name, an investigation is always about exactly one company, so every
    call site must be explicit about which one rather than silently
    searching across every persisted company's filings.
    """
    graph = _build_graph(
        session, chat_client, search_depth=search_depth, context_pages=context_pages
    )
    result = await graph.ainvoke(
        {"question": question, "company_number": company_number}
    )
    return cast(Finding, result["finding"])

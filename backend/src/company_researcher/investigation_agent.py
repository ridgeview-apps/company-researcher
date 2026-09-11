import re
from collections.abc import Sequence
from datetime import date
from typing import Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import trace
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import DocumentPage
from company_researcher.discriminative_query import derive_discriminative_query
from company_researcher.fiscal_year_extraction import extract_fiscal_years
from company_researcher.fiscal_year_lookup import (
    document_extraction_ids_for_fiscal_year,
)
from company_researcher.human_review import needs_human_review, record_pending_review
from company_researcher.lexical_search import (
    PageMatch,
    search_pages,
    text_matches_query,
)
from company_researcher.llm_client import ChatMessage, ChatUsage, UsageAwareChatProvider

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

_CLAIM_TYPE_INSTRUCTION = (
    "Every answer must also be classified with claim_type, either 'fact' or "
    "'interpretation'. Use 'fact' when the claim states only what the "
    "evidence directly says (e.g. 'three directors resigned within 14 "
    "months'). Use 'interpretation' when the claim adds a judgement that "
    "goes beyond what the evidence directly states (e.g. 'this indicates "
    "governance instability') - even if that judgement seems reasonable. "
    "An interpretation is not wrong to offer, but must always be labelled "
    "as one rather than presented as a directly evidenced fact."
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
    "discuss the same topic. " + _CLAIM_TYPE_INSTRUCTION
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
    "comparison, set evidence_sufficient to false and say so. "
    + _CLAIM_TYPE_INSTRUCTION
)


_CLAIM_TYPE_RECLASSIFICATION_SYSTEM_PROMPT = (
    "You are checking whether a claim actually answers a question, using "
    "only the question and the claim - no evidence text, so you cannot "
    "judge whether the claim is true, only whether its own wording, read "
    "against the question, is a direct statement of what was found (fact) "
    "or goes beyond that into a judgement, inference, or assessment of "
    "significance (interpretation). A claim that declines to answer an "
    "evaluative part of the question and instead only restates an "
    "underlying fact must still be classified as 'interpretation' if the "
    "question itself asks for a judgement (for example 'does X indicate "
    "Y', 'does X suggest Y', 'is X significant') and the claim does not "
    "actually render that judgement - presenting an incomplete answer as a "
    "settled fact is itself not something that should be treated as fully "
    "settled without human review. Respond with a claim_type of exactly "
    "'fact' or 'interpretation', and a one-sentence reason."
)


class InvestigationAgentError(Exception):
    """Raised when the agent produces a finding that violates its evidence contract."""


class ClaimTypeReclassification(BaseModel):
    """An independent, evidence-blind re-check of a finding's self-reported claim_type."""

    model_config = ConfigDict(extra="forbid")

    claim_type: Literal["fact", "interpretation"]
    reason: str


def _force_unambiguous_fiscal_year(query: str, question: str) -> str:
    """Append the question's fiscal year to `query` when exactly one is named."""
    years = extract_fiscal_years(question)
    # Skip multi-year questions: their hand-tuned queries deliberately omit
    # a year token (see build-log.md, "fixing the fiscal-year leak").
    if len(years) != 1:
        return query
    year = years[0]
    if re.search(rf"\b{year}\b", query):
        return query
    # generate_query's LLM doesn't reliably include the literal year token
    # that ts_rank needs to disambiguate near-duplicate year-over-year filings.
    return f"{query} {year}".strip()


def _fiscal_year_range(years: Sequence[str]) -> list[str]:
    """Expand 2+ named years into the inclusive range between the earliest and latest."""
    # A question only names its boundary years (e.g. "FY2021 through
    # FY2025" -> "2021", "2025"), but evidence is expected from every year
    # in between too. Single-year/no-year cases stay on the single-pass path.
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
    claim_type: Literal["fact", "interpretation"]
    evidence_sufficient: bool
    citations: list[Citation]


# Kept separate per year so each sub-finding is synthesized from only that
# year's own retrieved pages, avoiding one shared, mixed-year context window.
class YearEvidence(BaseModel):
    """One fiscal year's independently retrieved evidence and grounded sub-finding."""

    fiscal_year: str
    retrieved_pages: list[RetrievedPage]
    finding: Finding


# Narrower than InvestigationState (only the 3 fields ainvoke() is actually
# called with) so InvestigationState can require every field below without
# making this initial call fail to type-check.
class InvestigationInput(TypedDict):
    """The three fields actually supplied to `graph.ainvoke()` at the start of a run."""

    question: str
    company_number: str
    as_of_date: date | None


# Every field is required even though each node only returns a partial
# update - LangGraph merges each node's dict into this state - so node
# functions return dict[str, object] and are trusted, per the graph's edge
# ordering (not the type checker), to populate a field before it's read.
class InvestigationState(TypedDict):
    """LangGraph state threaded through the investigation graph."""

    question: str
    company_number: str
    as_of_date: date | None
    generated_query: str
    fiscal_year: str | None
    fiscal_year_range: list[str]
    retrieved_pages: list[RetrievedPage]
    year_evidence: list[YearEvidence]
    finding: Finding
    review_id: int | None
    usage_records: list[ChatUsage]


def _sum_usage(records: Sequence[ChatUsage]) -> ChatUsage | None:
    """Sum token usage across every LLM call an investigation made."""
    if not records:
        # None, not a zero-valued ChatUsage - a real run with zero cost is
        # otherwise indistinguishable from no usage being reported at all.
        return None
    return ChatUsage(
        prompt_tokens=sum(record.prompt_tokens for record in records),
        completion_tokens=sum(record.completion_tokens for record in records),
        total_tokens=sum(record.total_tokens for record in records),
    )


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
    """Strip whitespace/punctuation noise and case so a genuine quote isn't rejected for it."""
    # Real runs surfaced repeated non-substantive OCR/formatting noise that
    # would otherwise fail a genuine quote: "." vs "," as a thousands
    # separator, mismatched brackets ("{" for "("), a dropped space inside a
    # name, a newline-separated list joined into prose with an added
    # period, stray "©" / mid-word line-wrap hyphens from watermark
    # artifacts, ":" vs "." as a decimal point, and a stray "»" mid-sentence.
    # None of these change a word or digit sequence - only whitespace and
    # punctuation - so this is deliberately permissive (two different
    # numbers or adjacent unrelated words could in principle collide once
    # separators are stripped); it only checks quote *fidelity* to real page
    # text, not the claim's numeric/semantic correctness. Full case-by-case
    # history: build-log.md, "Verifying citation quotes".
    normalized = text.replace("{", "(").replace("}", ")")
    for character in (",", ".", "_", "-", "©", ":", "»"):
        normalized = normalized.replace(character, "")
    return "".join(normalized.split()).lower()


def _find_quote_mismatches(
    finding: Finding, retrieved_pages: Sequence[RetrievedPage]
) -> list[Citation]:
    """Return citations whose supporting_text is not a verbatim excerpt of its cited page."""
    # Assumes _validate_citations already confirmed every citation's page
    # was retrieved. Catches a citation pointing at a real page but quoting
    # text never actually written there (including text spliced together
    # from different parts of the page) - see build-log.md, "Verifying
    # citation quotes".
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
    chat_client: UsageAwareChatProvider,
    system_prompt: str,
    user_message: str,
    retrieved_pages: Sequence[RetrievedPage],
) -> tuple[Finding, list[ChatUsage]]:
    """Run one structured synthesis call and enforce both citation guarantees."""
    messages = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_message),
    ]
    finding, usage = await chat_client.complete_structured_with_usage(messages, Finding)
    usage_records = [usage] if usage is not None else []
    # Existence check: fail-closed, no retry - citing a page never retrieved
    # is a more severe error than an imprecise quote.
    _validate_citations(finding, retrieved_pages)
    mismatches = _find_quote_mismatches(finding, retrieved_pages)
    if not mismatches:
        return finding, usage_records

    # Quote-fidelity check: give the model one self-correction retry, naming
    # exactly which quote was wrong, before treating it as a hard failure.
    retry_message = f"{user_message}\n\n{_format_quote_correction_request(mismatches)}"
    retried_finding, retry_usage = await chat_client.complete_structured_with_usage(
        [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=retry_message),
        ],
        Finding,
    )
    if retry_usage is not None:
        usage_records.append(retry_usage)
    _validate_citations(retried_finding, retrieved_pages)
    remaining_mismatches = _find_quote_mismatches(retried_finding, retrieved_pages)
    if remaining_mismatches:
        citation = remaining_mismatches[0]
        raise InvestigationAgentError(
            f"Finding cited document_extraction_id={citation.document_extraction_id} "
            f"page_number={citation.page_number} with a supporting_text quote that is "
            "not verbatim text from that page, even after a self-correction retry"
        )
    return retried_finding, usage_records


async def _apply_evidence_relevance_backstop(
    session: AsyncSession, question: str, finding: Finding
) -> Finding:
    """Force evidence_sufficient=False when no citation shares any discriminative term with the question."""
    if not finding.evidence_sufficient:
        return finding
    # Closes a real adversarial-injection gap: a page unrelated to the
    # question could still be cited as if it answered it, with the model
    # self-reporting evidence_sufficient=True regardless. Reuses
    # derive_discriminative_query's corpus document-frequency ranking
    # (already built for retrieval) rather than matching on any content
    # word, which would false-positive on boilerplate nearly every page
    # shares (e.g. "company"). See build-log.md, "Adversarial /
    # prompt-injection testing".
    discriminative_query = await derive_discriminative_query(session, question)
    if not discriminative_query:
        return finding
    # Checked against each citation's own supporting_text, not the full
    # retrieved page: the page may contain unrelated or injected content
    # that mentions the question's terms without the citation relying on it.
    citation_text = " ".join(
        citation.supporting_text for citation in finding.citations
    ).strip()
    if citation_text and await text_matches_query(
        session, citation_text, discriminative_query
    ):
        return finding
    # Only ever narrows True -> False, never the reverse.
    return finding.model_copy(update={"evidence_sufficient": False})


_JUDGEMENT_SEEKING_PHRASES = (
    "indicate",
    "suggest",
    "imply",
    "sign of",
    "reflect",
    "consistent with",
    "raise concerns",
    "raise questions",
    "does this mean",
)


def _question_seeks_judgement(question: str) -> bool:
    """Detect, deterministically, whether a question itself asks for a judgement."""
    # A separate technique from _reclassify_claim_type, not a variant of it:
    # that LLM call failed 4 consecutive adversarial-testing runs against an
    # injected page baiting the model into a bare factual recitation that
    # technically dodges an evaluative question - the reclassifier, reading
    # only that evasive claim, never recognized the evasion as needing
    # 'interpretation'. This check reads only the user's own question text,
    # never the claim or any evidence-derived content, so no instruction
    # embedded in a retrieved page can reach it. A fixed-phrase heuristic,
    # not a semantic parser - see build-log.md, "Closing the HITL-bypass gap".
    lowered = question.lower()
    return any(phrase in lowered for phrase in _JUDGEMENT_SEEKING_PHRASES)


def _apply_question_judgement_backstop(question: str, finding: Finding) -> Finding:
    """Force claim_type=interpretation when the question itself asks for a judgement."""
    if finding.claim_type == "interpretation":
        return finding
    if not _question_seeks_judgement(question):
        return finding
    # Only ever upgrades fact -> interpretation, never the reverse.
    return finding.model_copy(update={"claim_type": "interpretation"})


async def _reclassify_claim_type(
    chat_client: UsageAwareChatProvider, question: str, finding: Finding
) -> tuple[Finding, ChatUsage | None]:
    """Re-check a self-reported claim_type="fact" with a call that never sees evidence text."""
    # Closes the other adversarial-injection gap: an instruction embedded in
    # a retrieved page can bait synthesis into self-labelling an
    # interpretation (or an evasive answer) as "fact". This second call only
    # ever sees the question and the already-produced claim, never
    # evidence-derived text, so no hidden instruction can reach it. Only
    # called when self-reported as "fact", and only ever upgrades fact ->
    # interpretation - a backstop against under-flagging, not a
    # general-purpose reclassifier that could itself suppress review.
    if finding.claim_type == "interpretation":
        return finding, None
    user_message = f"Question: {question}\n\nClaim: {finding.claim}"
    reclassification, usage = await chat_client.complete_structured_with_usage(
        [
            ChatMessage(
                role="system", content=_CLAIM_TYPE_RECLASSIFICATION_SYSTEM_PROMPT
            ),
            ChatMessage(role="user", content=user_message),
        ],
        ClaimTypeReclassification,
    )
    if reclassification.claim_type == finding.claim_type:
        return finding, usage
    return finding.model_copy(update={"claim_type": reclassification.claim_type}), usage


async def _apply_review_integrity_checks(
    session: AsyncSession,
    chat_client: UsageAwareChatProvider,
    question: str,
    finding: Finding,
) -> tuple[Finding, list[ChatUsage]]:
    """Apply all three human-review-gate backstops to a synthesized finding."""
    # All three only ever push toward requiring review, never away from it -
    # a deliberately asymmetric safety net against the self-classification
    # manipulation adversarial-injection testing found.
    finding = await _apply_evidence_relevance_backstop(session, question, finding)
    # Runs before _reclassify_claim_type deliberately: it's free (no LLM
    # call), and when it already upgrades to 'interpretation', the LLM
    # call's early-return skips entirely - this ordering can only cut cost.
    finding = _apply_question_judgement_backstop(question, finding)
    finding, usage = await _reclassify_claim_type(chat_client, question, finding)
    return finding, [usage] if usage is not None else []


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
    """Render each year's already-grounded sub-finding for the aggregation prompt."""
    # Only claim/sufficiency/citations, not raw page text again - grounding
    # already happened once per year; aggregation is a narrative/comparison
    # layer over already-validated facts.
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
    chat_client: UsageAwareChatProvider,
    *,
    search_depth: int,
    context_pages: int,
) -> CompiledStateGraph[
    InvestigationState, None, InvestigationInput, InvestigationState
]:
    """Assemble the investigation graph."""
    # generate_query always runs first, then branches on how many fiscal
    # years the question names: 0 or 1 takes the original single
    # retrieve_evidence -> synthesize_finding pass; 2+ (a genuinely
    # multi-year question) takes a per-year gather_year_findings ->
    # aggregate_findings pass instead, so one year's evidence never crowds
    # out another's in a single shared context window.

    async def generate_query_node(state: InvestigationState) -> dict[str, object]:
        query, usage = await chat_client.complete_with_usage(
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
            "usage_records": [usage] if usage is not None else [],
        }

    async def retrieve_evidence_node(state: InvestigationState) -> dict[str, object]:
        fiscal_year = state.get("fiscal_year")
        document_extraction_ids = None
        if fiscal_year is not None:
            candidate_ids = await document_extraction_ids_for_fiscal_year(
                session, fiscal_year, company_number=state["company_number"]
            )
            # Empty can mean a genuine reporting gap, or the named year
            # wasn't actually an accounting period (e.g. a charge-creation
            # date) - this lookup can't tell the two apart. Falling back to
            # no restriction (rather than matching zero pages) accepts a
            # small over-broad-retrieval risk instead of failing closed on
            # an answerable question (observed on a real Nothing Technology
            # run). Deliberately narrower than gather_year_findings_node's
            # multi-year path, which must keep reporting
            # evidence_sufficient=False for a genuinely absent year rather
            # than silently widen. Scoped by company_number so "empty"
            # means this company lacks the filing, not the whole corpus.
            document_extraction_ids = candidate_ids or None
        matches = await search_pages(
            session,
            state["generated_query"],
            limit=search_depth,
            document_extraction_ids=document_extraction_ids,
            company_number=state["company_number"],
            as_of_date=state.get("as_of_date"),
        )
        pages = await _load_page_texts(session, matches[:context_pages])
        return {"retrieved_pages": pages}

    async def synthesize_finding_node(state: InvestigationState) -> dict[str, object]:
        pages = state["retrieved_pages"]
        evidence_text = _format_evidence_text(
            pages, empty_message="No evidence pages were retrieved for this question."
        )
        user_message = f"Question: {state['question']}\n\nAvailable evidence pages:\n\n{evidence_text}"
        finding, usage_records = await _synthesize_and_validate(
            chat_client, _FINDING_SYSTEM_PROMPT, user_message, pages
        )
        finding, integrity_usage = await _apply_review_integrity_checks(
            session, chat_client, state["question"], finding
        )
        return {
            "finding": finding,
            "usage_records": state.get("usage_records", [])
            + usage_records
            + integrity_usage,
        }

    async def gather_year_findings_node(
        state: InvestigationState,
    ) -> dict[str, object]:
        question = state["question"]
        query = state["generated_query"]
        year_evidence: list[YearEvidence] = []
        all_usage_records: list[ChatUsage] = []
        # Each year gets its own LangSmith span (a no-op when tracing is off)
        # so this loop's per-year decomposition - otherwise one opaque node
        # call - is visible as sibling spans in a trace.
        for year in state["fiscal_year_range"]:
            async with trace(
                f"fiscal_year_{year}", run_type="chain", inputs={"fiscal_year": year}
            ) as year_run:
                document_extraction_ids = await document_extraction_ids_for_fiscal_year(
                    session, year, company_number=state["company_number"]
                )
                matches = await search_pages(
                    session,
                    query,
                    limit=search_depth,
                    document_extraction_ids=document_extraction_ids,
                    company_number=state["company_number"],
                    as_of_date=state.get("as_of_date"),
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
                finding, usage_records = await _synthesize_and_validate(
                    chat_client, _FINDING_SYSTEM_PROMPT, user_message, pages
                )
                finding, integrity_usage = await _apply_review_integrity_checks(
                    session, chat_client, question, finding
                )
                all_usage_records.extend(usage_records)
                all_usage_records.extend(integrity_usage)
                year_evidence.append(
                    YearEvidence(
                        fiscal_year=year, retrieved_pages=pages, finding=finding
                    )
                )
                year_run.end(
                    outputs={
                        "retrieved_page_count": len(pages),
                        "claim": finding.claim,
                        "evidence_sufficient": finding.evidence_sufficient,
                    }
                )
        return {
            "year_evidence": year_evidence,
            "usage_records": state.get("usage_records", []) + all_usage_records,
        }

    async def aggregate_findings_node(state: InvestigationState) -> dict[str, object]:
        year_evidence = state["year_evidence"]
        summary = _format_year_findings_summary(year_evidence)
        user_message = (
            f"Question: {state['question']}\n\nPer-year findings:\n\n{summary}"
        )
        all_pages = [
            page for evidence in year_evidence for page in evidence.retrieved_pages
        ]
        finding, usage_records = await _synthesize_and_validate(
            chat_client, _AGGREGATE_SYSTEM_PROMPT, user_message, all_pages
        )
        finding, integrity_usage = await _apply_review_integrity_checks(
            session, chat_client, state["question"], finding
        )
        return {
            "finding": finding,
            "usage_records": state.get("usage_records", [])
            + usage_records
            + integrity_usage,
        }

    async def human_review_gate_node(state: InvestigationState) -> dict[str, object]:
        finding = state["finding"]
        if not needs_human_review(
            claim_type=finding.claim_type,
            evidence_sufficient=finding.evidence_sufficient,
        ):
            return {"review_id": None}
        review_id = await record_pending_review(
            session,
            company_number=state["company_number"],
            question=state["question"],
            generated_query=state["generated_query"],
            claim=finding.claim,
            claim_type=finding.claim_type,
            evidence_sufficient=finding.evidence_sufficient,
            citations=[citation.model_dump() for citation in finding.citations],
        )
        return {"review_id": review_id}

    graph = StateGraph(InvestigationState, input_schema=InvestigationInput)
    graph.add_node("generate_query", generate_query_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("synthesize_finding", synthesize_finding_node)
    graph.add_node("gather_year_findings", gather_year_findings_node)
    graph.add_node("aggregate_findings", aggregate_findings_node)
    graph.add_node("human_review_gate", human_review_gate_node)
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
    graph.add_edge("synthesize_finding", "human_review_gate")
    graph.add_edge("gather_year_findings", "aggregate_findings")
    graph.add_edge("aggregate_findings", "human_review_gate")
    graph.add_edge("human_review_gate", END)
    return graph.compile()


async def _run_graph(
    session: AsyncSession,
    chat_client: UsageAwareChatProvider,
    question: str,
    company_number: str,
    *,
    search_depth: int,
    context_pages: int,
    as_of_date: date | None,
) -> InvestigationState:
    """Build and run the investigation graph, returning its final state."""
    # Shared by investigate(), investigate_with_review(), and
    # investigate_with_usage() so the three differ only in what they read
    # out of the final state, not in how the graph is built or invoked.
    graph = _build_graph(
        session, chat_client, search_depth=search_depth, context_pages=context_pages
    )
    return cast(
        InvestigationState,
        await graph.ainvoke(
            {
                "question": question,
                "company_number": company_number,
                "as_of_date": as_of_date,
            }
        ),
    )


async def investigate(
    session: AsyncSession,
    chat_client: UsageAwareChatProvider,
    question: str,
    company_number: str,
    *,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
    context_pages: int = DEFAULT_CONTEXT_PAGES,
    as_of_date: date | None = None,
) -> Finding:
    """Run the investigation graph for one natural-language question."""
    # Lexical search only: hand-tuned lexical outperforms both vector-only
    # and naive hybrid on this project's measured corpus (see README.md's
    # "At a glance"), so lexical is what this agent calls. Unlike the
    # evaluation dataset's hand-tuned queries, generated_query is produced
    # by the LLM from the question alone, at run time.
    result = await _run_graph(
        session,
        chat_client,
        question,
        company_number,
        search_depth=search_depth,
        context_pages=context_pages,
        as_of_date=as_of_date,
    )
    return result["finding"]


async def investigate_with_review(
    session: AsyncSession,
    chat_client: UsageAwareChatProvider,
    question: str,
    company_number: str,
    *,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
    context_pages: int = DEFAULT_CONTEXT_PAGES,
    as_of_date: date | None = None,
) -> tuple[Finding, int | None]:
    """Run one investigation and also return a pending human review ID, if one was raised."""
    # A separate function rather than changing investigate()'s return
    # contract, so every existing caller (the CLI, and tests calling
    # investigate() directly) is unaffected. human_review_gate decides
    # whether review is needed and persists it; this just reads that
    # decision back out of the final graph state. None means no review needed.
    result = await _run_graph(
        session,
        chat_client,
        question,
        company_number,
        search_depth=search_depth,
        context_pages=context_pages,
        as_of_date=as_of_date,
    )
    return result["finding"], result.get("review_id")


async def investigate_with_usage(
    session: AsyncSession,
    chat_client: UsageAwareChatProvider,
    question: str,
    company_number: str,
    *,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
    context_pages: int = DEFAULT_CONTEXT_PAGES,
    as_of_date: date | None = None,
) -> tuple[Finding, ChatUsage | None]:
    """Run one investigation and also return its total token usage."""
    # Separate from investigate() for the same reason investigate_with_review()
    # is: existing callers stay unaffected. Sums usage across every LLM call
    # in the run (query generation, every synthesis call including retries,
    # and - for a multi-year question - every per-year pass plus aggregation).
    result = await _run_graph(
        session,
        chat_client,
        question,
        company_number,
        search_depth=search_depth,
        context_pages=context_pages,
        as_of_date=as_of_date,
    )
    usage = _sum_usage(result.get("usage_records", []))
    return result["finding"], usage

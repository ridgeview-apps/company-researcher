from collections.abc import AsyncIterator, Callable, Sequence
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
    HumanReview,
)
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.investigation_agent import (
    Citation,
    ClaimTypeReclassification,
    Finding,
    InvestigationAgentError,
    _apply_evidence_relevance_backstop,
    _apply_question_judgement_backstop,
    _apply_review_integrity_checks,
    _force_unambiguous_fiscal_year,
    _normalize_for_quote_check,
    _question_seeks_judgement,
    _reclassify_claim_type,
    investigate,
    investigate_with_review,
    investigate_with_usage,
)
from company_researcher.llm_client import ChatMessage, ChatUsage

TEST_COMPANY_NUMBER = "TE000008"

_StructuredResponse = TypeVar("_StructuredResponse", bound=BaseModel)


class FakeChatClient:
    """Returns a fixed query for `complete_with_usage`.

    `complete_structured_with_usage` returns `finding` on every call
    unless `finding_selector` is given, in which case it picks a response
    by inspecting each call's messages - needed for multi-year tests,
    where it is called once per fiscal year plus once more to aggregate,
    each expecting a different fake response. `usage` defaults to `None`
    since most tests exercise `investigate()`'s behavior, not token
    accounting; pass a fixed `ChatUsage` to also verify accumulation via
    `investigate_with_usage`. `reclassification`, when given, is returned
    for every independent claim_type reclassification call (see
    `_reclassify_claim_type` in investigation_agent.py); it defaults to
    agreeing with a self-reported claim_type="fact", so existing tests that
    don't care about this backstop are unaffected - that call is only ever
    made when a finding's own claim_type is "fact", so an
    interpretation-classified finding's fixture never needs one.
    """

    def __init__(
        self,
        *,
        query: str,
        finding: Finding | None = None,
        finding_selector: Callable[[Sequence[ChatMessage]], Finding] | None = None,
        usage: ChatUsage | None = None,
        reclassification: ClaimTypeReclassification | None = None,
    ) -> None:
        self._query = query
        self._finding = finding
        self._finding_selector = finding_selector
        self._usage = usage
        self._reclassification = reclassification
        self.complete_calls: list[Sequence[ChatMessage]] = []
        self.complete_structured_calls: list[Sequence[ChatMessage]] = []

    async def complete_with_usage(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[str, ChatUsage | None]:
        self.complete_calls.append(messages)
        return self._query, self._usage

    async def complete_structured_with_usage(
        self, messages: Sequence[ChatMessage], response_model: type[_StructuredResponse]
    ) -> tuple[_StructuredResponse, ChatUsage | None]:
        self.complete_structured_calls.append(messages)
        if response_model is ClaimTypeReclassification:
            reclassification = self._reclassification or ClaimTypeReclassification(
                claim_type="fact", reason="test default: agrees with the finding as-is"
            )
            return cast(_StructuredResponse, reclassification), self._usage
        if self._finding_selector is not None:
            return cast(
                _StructuredResponse, self._finding_selector(messages)
            ), self._usage
        assert self._finding is not None
        return cast(_StructuredResponse, self._finding), self._usage


def _finding_selector_by_year(
    per_year: dict[str, Finding], aggregate: Finding
) -> Callable[[Sequence[ChatMessage]], Finding]:
    """Route each `complete_structured` call to its matching fake response.

    Per-year calls are identified by the "Focus specifically on fiscal
    year {year}" marker `gather_year_findings_node` puts in its user
    message; the final aggregation call is identified by its own distinct
    "Per-year findings:" marker.
    """

    def selector(messages: Sequence[ChatMessage]) -> Finding:
        content = messages[-1].content
        if "Per-year findings:" in content:
            return aggregate
        for year, finding in per_year.items():
            if f"fiscal year {year}" in content:
                return finding
        raise AssertionError(f"No fake finding configured for prompt: {content!r}")

    return selector


_RETRY_MARKER = "was not an exact, contiguous quote copied verbatim"


def _finding_selector_with_retry(
    initial: Finding, corrected: Finding
) -> Callable[[Sequence[ChatMessage]], Finding]:
    """Return `initial` on the first `complete_structured` call, `corrected` on the retry.

    `_synthesize_and_validate`'s retry prompt always contains `_RETRY_MARKER`
    (from `_format_quote_correction_request`), so its presence distinguishes
    the retry call from the original one regardless of which node issued it.
    """

    def selector(messages: Sequence[ChatMessage]) -> Finding:
        content = messages[-1].content
        return corrected if _RETRY_MARKER in content else initial

    return selector


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


def test_normalize_for_quote_check_tolerates_ocr_digit_separator_confusion() -> None:
    """Regression test for a real observed failure: OCR renders a thousands separator as '.' instead of ',' (e.g. "437.629" for "437,629")."""
    assert _normalize_for_quote_check("437.629") == _normalize_for_quote_check(
        "437,629"
    )


def test_normalize_for_quote_check_tolerates_mismatched_brackets() -> None:
    """Regression test for a real observed failure: OCR renders a bracket pair with mismatched characters (e.g. an opening curly brace paired with a closing parenthesis)."""
    assert _normalize_for_quote_check(
        "{Appointed 9 January 2023)"
    ) == _normalize_for_quote_check("(Appointed 9 January 2023)")


def test_normalize_for_quote_check_strips_underscore_leaders() -> None:
    assert _normalize_for_quote_check("__260.674") == _normalize_for_quote_check(
        "260,674"
    )


def test_normalize_for_quote_check_tolerates_a_newline_list_quoted_as_prose() -> None:
    """Regression test for a real observed failure: the page lists items one per line (e.g. a list of directors), but the model naturally quotes them as a comma-separated, period-terminated sentence."""
    page = "The directors who served during the year were:\n\nB Francis\n\nS Hewitt\n\nP Daw"
    quote = "The directors who served during the year were: B Francis, S Hewitt, P Daw."
    assert _normalize_for_quote_check(quote) in _normalize_for_quote_check(page)


def test_normalize_for_quote_check_tolerates_a_missing_space_inside_a_name() -> None:
    """Regression test for a real observed failure: OCR drops a space inside a name ("N AMcElhinney"), but the model naturally quotes it correctly spaced ("N A McElhinney")."""
    assert _normalize_for_quote_check("N AMcElhinney") == _normalize_for_quote_check(
        "N A McElhinney"
    )


def test_normalize_for_quote_check_still_distinguishes_different_numbers() -> None:
    """The punctuation tolerance above must not become so permissive that genuinely different figures collapse together."""
    assert _normalize_for_quote_check("437,629") != _normalize_for_quote_check(
        "500,000"
    )


def test_normalize_for_quote_check_tolerates_a_colon_in_place_of_a_decimal_point() -> (
    None
):
    """Regression test for a real observed failure on Nothing Technology's corpus: OCR renders a ':' instead of '.' as a monetary decimal point (e.g. "£43:4m" for "£43.4m"), which caused a real citation quoting the correct figure to fail verification."""
    assert _normalize_for_quote_check("£43.4m") == _normalize_for_quote_check("£43:4m")


def test_normalize_for_quote_check_tolerates_a_stray_guillemet() -> None:
    """Regression test for a real observed failure on Nothing Technology's corpus: the real page reads "amounted to £59.4m » (2022: loss of £43:4m)", where OCR inserts a stray '»' character the model naturally omits as meaningless noise when quoting - confirmed by re-running the exact question that motivated the colon fix above and finding it still failed on this same sentence for this separate reason."""
    page = "The loss for the period, after taxation, amounted to £59.4m » (2022: loss of £43:4m)."
    quote = "The loss for the period, after taxation, amounted to £59.4m (2022: loss of £43.4m)."
    assert _normalize_for_quote_check(quote) in _normalize_for_quote_check(page)


def test_normalize_for_quote_check_tolerates_a_stray_symbol_at_a_linewrap_hyphen() -> (
    None
):
    """Regression test for a real observed failure on Nothing Technology's corpus: OCR inserts a stray '©' character and a line-wrap hyphen (e.g. "debt ©\n-fundraising" for "debt fundraising"), from a PDF carrying DocuSign watermark artifacts Gymshark's filings did not have."""
    page = "a £30m debt ©\n-fundraising in order to support working capital"
    quote = "a £30m debt fundraising in order to support working capital"
    assert _normalize_for_quote_check(quote) in _normalize_for_quote_check(page)


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
    filing_date: date = date(2026, 1, 1),
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
        date=filing_date,
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
async def test_apply_evidence_relevance_backstop_forces_false_for_unrelated_citation(
    session: AsyncSession,
) -> None:
    """Regression test for the real adversarial-injection finding (see README's
    "Adversarial / prompt-injection testing" section): a citation that is
    genuinely, topically unrelated to the question must not be allowed to
    support a confident evidence_sufficient=True claim.
    """
    finding = Finding(
        claim="This filing does not provide evidence of a fraud investigation.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=1,
                page_number=1,
                supporting_text="the company continued to trade in its principal activity of retail distribution",
            )
        ],
    )

    result = await _apply_evidence_relevance_backstop(
        session,
        "Does this filing provide evidence that the company was under investigation for fraud?",
        finding,
    )

    assert result.evidence_sufficient is False


@pytest.mark.asyncio
async def test_apply_evidence_relevance_backstop_leaves_a_related_citation_unchanged(
    session: AsyncSession,
) -> None:
    finding = Finding(
        claim="The directors consider the going concern basis appropriate.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=1,
                page_number=1,
                supporting_text="the directors consider the going concern basis to be appropriate",
            )
        ],
    )

    result = await _apply_evidence_relevance_backstop(
        session, "What is the going concern position?", finding
    )

    assert result.evidence_sufficient is True


@pytest.mark.asyncio
async def test_apply_evidence_relevance_backstop_never_upgrades_already_insufficient(
    session: AsyncSession,
) -> None:
    finding = Finding(
        claim="Insufficient evidence to answer.",
        claim_type="fact",
        evidence_sufficient=False,
        citations=[],
    )

    result = await _apply_evidence_relevance_backstop(
        session, "Anything unrelated to this corpus?", finding
    )

    assert result.evidence_sufficient is False


def test_question_seeks_judgement_detects_the_exploited_pattern() -> None:
    """Regression test for the real, previously-failing adversarial case
    (`interpretation-bait-governance-instability`): the reclassifier LLM
    call failed 4 consecutive real runs to recognize this question as
    seeking a judgement, because the injected claim it read was an
    evasive-but-technically-correct factual recitation. This check reads
    only the question, never the claim, so it does not depend on the
    claim's own wording at all.
    """
    assert _question_seeks_judgement(
        "Do the resignations disclosed in this filing indicate governance instability?"
    )


def test_question_seeks_judgement_is_false_for_an_ordinary_factual_question() -> None:
    assert not _question_seeks_judgement("What was Gymshark's turnover in FY2023?")


@pytest.mark.asyncio
async def test_apply_question_judgement_backstop_upgrades_a_judgement_seeking_question() -> (
    None
):
    finding = Finding(
        claim="The following directors resigned during the year: J Smith, R Patel.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[],
    )

    result = _apply_question_judgement_backstop(
        "Do the resignations disclosed in this filing indicate governance instability?",
        finding,
    )

    assert result.claim_type == "interpretation"
    assert result.claim == finding.claim


def test_apply_question_judgement_backstop_leaves_an_ordinary_fact_unchanged() -> None:
    finding = Finding(
        claim="Turnover was £490,142k.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[],
    )

    result = _apply_question_judgement_backstop(
        "What was Gymshark's turnover in FY2023?", finding
    )

    assert result is finding


def test_apply_question_judgement_backstop_never_downgrades_an_interpretation() -> None:
    finding = Finding(
        claim="This indicates governance instability.",
        claim_type="interpretation",
        evidence_sufficient=True,
        citations=[],
    )

    result = _apply_question_judgement_backstop(
        "What was Gymshark's turnover in FY2023?", finding
    )

    assert result is finding


@pytest.mark.asyncio
async def test_apply_review_integrity_checks_skips_the_reclassifier_call_once_the_deterministic_backstop_already_upgraded(
    session: AsyncSession,
) -> None:
    """The deterministic question-judgement backstop runs before the LLM
    reclassifier, so when it already upgrades claim_type to
    'interpretation', `_reclassify_claim_type`'s own early-return means no
    LLM call happens at all - this ordering can only reduce cost, never
    add to it.
    """
    finding = Finding(
        claim="The following directors resigned during the year: J Smith, R Patel.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[],
    )
    chat_client = FakeChatClient(query="unused", finding=finding)

    result, usage_records = await _apply_review_integrity_checks(
        session,
        chat_client,
        "Do the resignations disclosed in this filing indicate governance instability?",
        finding,
    )

    assert result.claim_type == "interpretation"
    assert usage_records == []
    assert len(chat_client.complete_structured_calls) == 0


@pytest.mark.asyncio
async def test_reclassify_claim_type_upgrades_fact_to_interpretation_when_disagreed() -> (
    None
):
    """The independent reclassification call never sees evidence text, so an
    injected page instruction cannot reach it (see README's
    "Adversarial / prompt-injection testing" section).
    """
    finding = Finding(
        claim="The following directors resigned during the year: J Smith, R Patel.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[],
    )
    chat_client = FakeChatClient(
        query="unused",
        finding=finding,
        reclassification=ClaimTypeReclassification(
            claim_type="interpretation",
            reason="the claim only restates facts without answering the judgement asked",
        ),
    )

    result, _usage_records = await _reclassify_claim_type(
        chat_client, "Do these resignations indicate governance instability?", finding
    )

    assert result.claim_type == "interpretation"
    assert result.claim == finding.claim
    assert result.citations == finding.citations


@pytest.mark.asyncio
async def test_reclassify_claim_type_is_skipped_when_already_interpretation() -> None:
    """Only ever used to upgrade fact -> interpretation, never the reverse - already
    being an interpretation means no call is made at all, so it can never be
    downgraded back to fact by this check.
    """
    finding = Finding(
        claim="This indicates governance instability.",
        claim_type="interpretation",
        evidence_sufficient=True,
        citations=[],
    )
    chat_client = FakeChatClient(query="unused", finding=finding)

    result, usage_records = await _reclassify_claim_type(
        chat_client, "Does this indicate governance instability?", finding
    )

    assert result is finding
    assert usage_records is None
    assert len(chat_client.complete_structured_calls) == 0


@pytest.mark.asyncio
async def test_investigate_returns_a_citation_grounded_finding(
    session: AsyncSession, company: Company
) -> None:
    # `search_pages` is now scoped to `company` (TEST_COMPANY_NUMBER), so
    # this test's pages cannot collide with the real persisted Gymshark
    # corpus in the same shared development database purely through
    # company scoping. Fixture text and the fake query are still kept
    # deliberately distinctive nonsense, matching
    # test_retrieval_evaluation.py's convention, as defence in depth.
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
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                # Includes "alpha bravo charlie" (not just the narrower
                # "delta echo foxtrot" tail) so the citation's own text
                # shares a term with the question's own vocabulary -
                # required for the evidence-relevance backstop
                # (_apply_evidence_relevance_backstop) to recognize this as
                # on-topic, since the nonsense phrase this fixture uses is
                # rare enough in the corpus to dominate the question's
                # derived discriminative terms.
                supporting_text="Alpha bravo charlie identifies a delta echo foxtrot as evidence",
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
        company_number=TEST_COMPANY_NUMBER,
    )

    assert finding == expected_finding
    assert len(chat_client.complete_calls) == 1
    assert chat_client.complete_calls[0][-1] == ChatMessage(
        role="user",
        content="What did alpha bravo charlie identify as evidence?",
    )
    # 1 synthesis call, plus 1 claim_type reclassification call (see
    # `_apply_review_integrity_checks`).
    assert len(chat_client.complete_structured_calls) == 2
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
    chat_client = FakeChatClient(
        query="juliett kilo lima mike november oscar", finding=hallucinated_finding
    )

    with pytest.raises(InvestigationAgentError):
        await investigate(
            session,
            chat_client,
            "What did juliett kilo lima mention?",
            company_number=TEST_COMPANY_NUMBER,
        )


@pytest.mark.asyncio
async def test_investigate_reports_insufficient_evidence_when_nothing_is_retrieved(
    session: AsyncSession, company: Company
) -> None:
    insufficient_finding = Finding(
        claim="The retrieved evidence does not address this question.",
        claim_type="fact",
        evidence_sufficient=False,
        citations=[],
    )
    # A single unbroken nonsense token: PostgreSQL's text-search parser
    # splits on underscores/punctuation, so a phrase built from ordinary
    # words (even deliberately odd ones) risks a stray token matching real
    # corpus boilerplate. One fabricated token cannot match anything.
    chat_client = FakeChatClient(query="zqxvwkploqnhfbyt", finding=insufficient_finding)

    finding = await investigate(
        session,
        chat_client,
        "What is completely unrelated to this corpus?",
        company_number=TEST_COMPANY_NUMBER,
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
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=correct_extraction.id,
                page_number=1,
                supporting_text="Quebec romeo sierra tango whiskey xray disclosure for 2023",
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
        company_number=TEST_COMPANY_NUMBER,
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
        claim_type="fact",
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
            company_number=TEST_COMPANY_NUMBER,
            context_pages=2,
        )

    # The correct FY2023 filing's page must still be reachable.
    correct_finding = Finding(
        claim="Yankee zulu alpha beta gamma, per the 2023 filing.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=correct_extraction.id,
                page_number=1,
                supporting_text="Yankee zulu alpha beta gamma disclosure for the year ended 2023",
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
        company_number=TEST_COMPANY_NUMBER,
        context_pages=2,
    )

    assert finding == correct_finding


@pytest.mark.asyncio
async def test_investigate_excludes_a_filing_registered_after_the_as_of_cutoff(
    session: AsyncSession, company: Company
) -> None:
    """A filing publicly registered after `as_of_date` must be excluded from evidence entirely.

    Mirrors the real Gymshark original/amended FY2022 accounts pair: two
    filings share near-identical content, but one was registered with
    Companies House (`Filing.date`) after the cutoff and must not be
    reachable as evidence, even though `_validate_citations` would
    otherwise accept it as a real, retrieved page.
    """
    before_cutoff = await _create_filing_with_pages(
        session,
        "investigation-transaction-as-of-before",
        ["November papa quebec romeo disclosure, original filing."],
        filing_date=date(2023, 4, 22),
    )
    after_cutoff = await _create_filing_with_pages(
        session,
        "investigation-transaction-as-of-after",
        ["November papa quebec romeo disclosure, amended filing."],
        filing_date=date(2023, 11, 23),
    )
    hallucinated_finding = Finding(
        claim="Fabricated claim citing a filing registered after the cutoff.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=before_cutoff.id,
                page_number=1,
                supporting_text="November papa quebec romeo disclosure, original filing",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="november papa quebec romeo disclosure", finding=hallucinated_finding
    )

    # Sanity check: with no cutoff, both filings are reachable and the
    # "before" citation is accepted, confirming the fixture is set up
    # correctly and any later failure is really about the cutoff.
    finding = await investigate(
        session,
        chat_client,
        "What did november papa quebec romeo disclose?",
        company_number=TEST_COMPANY_NUMBER,
        context_pages=2,
    )
    assert finding == hallucinated_finding

    # A cutoff before the second filing's registration date must exclude it
    # from evidence entirely -- a citation to it is a validation failure,
    # not merely a low-ranked candidate.
    finding_citing_future_filing = Finding(
        claim="Fabricated claim citing the future-dated amended filing.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=after_cutoff.id,
                page_number=1,
                supporting_text="amended filing",
            )
        ],
    )
    chat_client_after_cutoff = FakeChatClient(
        query="november papa quebec romeo disclosure",
        finding=finding_citing_future_filing,
    )

    with pytest.raises(InvestigationAgentError):
        await investigate(
            session,
            chat_client_after_cutoff,
            "What did november papa quebec romeo disclose?",
            company_number=TEST_COMPANY_NUMBER,
            context_pages=2,
            as_of_date=date(2023, 9, 1),
        )


@pytest.mark.asyncio
async def test_investigate_as_of_cutoff_does_not_fall_back_when_nothing_qualifies(
    session: AsyncSession, company: Company
) -> None:
    """Unlike the fiscal-year restriction, an as-of cutoff must never fall back to unrestricted.

    A cutoff early enough to exclude every filing must report
    evidence_sufficient=False rather than silently widening the search --
    the entire point of this restriction is that a too-early cutoff finds
    nothing, not that it degrades gracefully into the fiscal-year
    fallback's behaviour.
    """
    await _create_filing_with_pages(
        session,
        "investigation-transaction-as-of-too-early",
        ["Sierra tango uniform victor disclosure, filed in 2026."],
        filing_date=date(2026, 1, 1),
    )
    insufficient_finding = Finding(
        claim="The retrieved evidence does not address this question.",
        claim_type="fact",
        evidence_sufficient=False,
        citations=[],
    )
    chat_client = FakeChatClient(
        query="sierra tango uniform victor disclosure", finding=insufficient_finding
    )

    finding = await investigate(
        session,
        chat_client,
        "What did sierra tango uniform victor disclose?",
        company_number=TEST_COMPANY_NUMBER,
        as_of_date=date(2020, 1, 1),
    )

    assert finding.evidence_sufficient is False
    assert finding.citations == []
    synthesis_prompt = chat_client.complete_structured_calls[0][-1].content
    assert "No evidence pages were retrieved" in synthesis_prompt


@pytest.mark.asyncio
async def test_investigate_falls_back_to_unrestricted_search_when_named_year_matches_no_filing(
    session: AsyncSession, company: Company
) -> None:
    """A named year that matches no filing's accounting period must not zero out retrieval entirely.

    Regression test for a real observed failure (see README.md): a
    question naming a year that refers to something other than an
    accounting period (e.g. a charge-creation date) retrieved nothing,
    because document_extraction_ids_for_fiscal_year correctly reports no
    filing has that accounting period, but search_pages treats an empty
    document_extraction_ids list as "match nothing" rather than "no
    restriction". The only filing in this test has a different accounting
    period than the year named in the question, so the fix must fall back
    to searching it unrestricted rather than returning zero pages.
    """
    extraction = await _create_filing_with_pages(
        session,
        "investigation-transaction-year-fallback",
        ["Tango uniform victor whiskey disclosure, dated in December 2024."],
        made_up_date="2023-07-31",
    )
    expected_finding = Finding(
        claim="Tango uniform victor whiskey was disclosed in December 2024.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="Tango uniform victor whiskey disclosure, dated in December 2024",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="tango uniform victor whiskey disclosure", finding=expected_finding
    )

    finding = await investigate(
        session,
        chat_client,
        "What did tango uniform victor whiskey disclose in December 2024?",
        company_number=TEST_COMPANY_NUMBER,
    )

    assert finding == expected_finding


@pytest.mark.asyncio
async def test_investigate_decomposes_a_multi_year_question_into_one_pass_per_year(
    session: AsyncSession, company: Company
) -> None:
    """A question naming 2+ years must gather evidence with one isolated pass per year, not one shared context window (see README's multi-step milestone).

    The same generated query matches every year's page (all three share the
    "cobalt zenith mosaic tundra" terms), but each year's retrieval is
    restricted to only that year's own filing, so the per-year prompts must
    each contain only their own year's figure, never another year's.
    """
    extraction_2021 = await _create_filing_with_pages(
        session,
        "investigation-multi-year-2021",
        ["Cobalt zenith mosaic tundra figure was 100 for 2021."],
        made_up_date="2021-07-31",
    )
    extraction_2022 = await _create_filing_with_pages(
        session,
        "investigation-multi-year-2022",
        ["Cobalt zenith mosaic tundra figure was 200 for 2022."],
        made_up_date="2022-07-31",
    )
    extraction_2023 = await _create_filing_with_pages(
        session,
        "investigation-multi-year-2023",
        ["Cobalt zenith mosaic tundra figure was 300 for 2023."],
        made_up_date="2023-07-31",
    )

    citation_2021 = Citation(
        document_extraction_id=extraction_2021.id,
        page_number=1,
        supporting_text="Cobalt zenith mosaic tundra figure was 100",
    )
    citation_2022 = Citation(
        document_extraction_id=extraction_2022.id,
        page_number=1,
        supporting_text="Cobalt zenith mosaic tundra figure was 200",
    )
    citation_2023 = Citation(
        document_extraction_id=extraction_2023.id,
        page_number=1,
        supporting_text="Cobalt zenith mosaic tundra figure was 300",
    )
    finding_2021 = Finding(
        claim="Cobalt zenith mosaic tundra figure was 100 in 2021.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[citation_2021],
    )
    finding_2022 = Finding(
        claim="Cobalt zenith mosaic tundra figure was 200 in 2022.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[citation_2022],
    )
    finding_2023 = Finding(
        claim="Cobalt zenith mosaic tundra figure was 300 in 2023.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[citation_2023],
    )
    aggregate_finding = Finding(
        claim="Cobalt zenith mosaic tundra rose from 100 in 2021 to 300 in 2023, via 200 in 2022.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[citation_2021, citation_2022, citation_2023],
    )
    chat_client = FakeChatClient(
        query="cobalt zenith mosaic tundra",
        finding_selector=_finding_selector_by_year(
            {"2021": finding_2021, "2022": finding_2022, "2023": finding_2023},
            aggregate_finding,
        ),
    )

    finding = await investigate(
        session,
        chat_client,
        "How did the cobalt zenith mosaic tundra figure change from 2021 to 2023?",
        company_number=TEST_COMPANY_NUMBER,
        context_pages=1,
    )

    assert finding == aggregate_finding
    assert len(chat_client.complete_calls) == 1
    # 4 synthesis calls (one per year plus the aggregate), each immediately
    # followed by its own independent claim_type reclassification call (see
    # `_apply_review_integrity_checks`) = 8 total.
    assert len(chat_client.complete_structured_calls) == 8

    prompt_2021 = chat_client.complete_structured_calls[0][-1].content
    prompt_2022 = chat_client.complete_structured_calls[2][-1].content
    prompt_2023 = chat_client.complete_structured_calls[4][-1].content
    assert "was 100" in prompt_2021
    assert "was 200" not in prompt_2021 and "was 300" not in prompt_2021
    assert "was 200" in prompt_2022
    assert "was 100" not in prompt_2022 and "was 300" not in prompt_2022
    assert "was 300" in prompt_2023
    assert "was 100" not in prompt_2023 and "was 200" not in prompt_2023

    aggregate_prompt = chat_client.complete_structured_calls[6][-1].content
    assert "Per-year findings:" in aggregate_prompt
    assert "figure was 100 in 2021" in aggregate_prompt
    # The aggregation prompt must reuse each year's already-grounded claim,
    # not re-dump raw OCR page text.
    assert (
        "Cobalt zenith mosaic tundra figure was 100 for 2021." not in aggregate_prompt
    )


@pytest.mark.asyncio
async def test_investigate_multi_year_gap_year_with_no_filing_still_gets_its_own_pass(
    session: AsyncSession, company: Company
) -> None:
    """A year in the named range with no persisted filing must still get its own gather pass, reporting insufficient evidence rather than being silently skipped."""
    extraction_2021 = await _create_filing_with_pages(
        session,
        "investigation-multi-year-gap-2021",
        ["Prairie glacier quartz reading was 50 for 2021."],
        made_up_date="2021-07-31",
    )
    # Deliberately no 2022 filing at all.
    extraction_2023 = await _create_filing_with_pages(
        session,
        "investigation-multi-year-gap-2023",
        ["Prairie glacier quartz reading was 70 for 2023."],
        made_up_date="2023-07-31",
    )

    citation_2021 = Citation(
        document_extraction_id=extraction_2021.id,
        page_number=1,
        supporting_text="Prairie glacier quartz reading was 50",
    )
    citation_2023 = Citation(
        document_extraction_id=extraction_2023.id,
        page_number=1,
        supporting_text="Prairie glacier quartz reading was 70",
    )
    finding_2021 = Finding(
        claim="Prairie glacier quartz reading was 50 in 2021.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[citation_2021],
    )
    finding_2022 = Finding(
        claim="No evidence of a 2022 prairie glacier quartz reading was found.",
        claim_type="fact",
        evidence_sufficient=False,
        citations=[],
    )
    finding_2023 = Finding(
        claim="Prairie glacier quartz reading was 70 in 2023.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[citation_2023],
    )
    aggregate_finding = Finding(
        claim=(
            "Prairie glacier quartz reading rose from 50 in 2021 to 70 in 2023; "
            "no 2022 filing was found."
        ),
        claim_type="fact",
        evidence_sufficient=True,
        citations=[citation_2021, citation_2023],
    )
    chat_client = FakeChatClient(
        query="prairie glacier quartz",
        finding_selector=_finding_selector_by_year(
            {"2021": finding_2021, "2022": finding_2022, "2023": finding_2023},
            aggregate_finding,
        ),
    )

    finding = await investigate(
        session,
        chat_client,
        "How did the prairie glacier quartz reading change from 2021 to 2023?",
        company_number=TEST_COMPANY_NUMBER,
        context_pages=1,
    )

    assert finding == aggregate_finding
    # 4 synthesis calls (one per year plus the aggregate), each immediately
    # followed by its own independent claim_type reclassification call (see
    # `_apply_review_integrity_checks`) = 8 total.
    assert len(chat_client.complete_structured_calls) == 8
    gap_year_prompt = chat_client.complete_structured_calls[2][-1].content
    assert "No evidence pages were retrieved for this fiscal year." in gap_year_prompt


@pytest.mark.asyncio
async def test_investigate_multi_year_rejects_a_sub_finding_that_cites_another_years_page(
    session: AsyncSession, company: Company
) -> None:
    """Each year's sub-finding must be validated against only that year's own retrieved pages, so a cross-year citation leak is still caught even inside the new multi-year path."""
    await _create_filing_with_pages(
        session,
        "investigation-multi-year-leak-2021",
        ["Marble copper vertex disclosure for 2021."],
        made_up_date="2021-07-31",
    )
    extraction_2022 = await _create_filing_with_pages(
        session,
        "investigation-multi-year-leak-2022",
        ["Marble copper vertex disclosure for 2022."],
        made_up_date="2022-07-31",
    )

    leaking_finding_2021 = Finding(
        claim="Fabricated claim citing the wrong year's page.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2022.id,
                page_number=1,
                supporting_text="disclosure for 2022",
            )
        ],
    )
    finding_2022 = Finding(
        claim="Marble copper vertex disclosure for 2022.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2022.id,
                page_number=1,
                supporting_text="disclosure for 2022",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="marble copper vertex",
        finding_selector=_finding_selector_by_year(
            {"2021": leaking_finding_2021, "2022": finding_2022},
            Finding(
                claim="unused",
                claim_type="fact",
                evidence_sufficient=True,
                citations=[],
            ),
        ),
    )

    with pytest.raises(InvestigationAgentError):
        await investigate(
            session,
            chat_client,
            "What did marble copper vertex disclose from 2021 to 2022?",
            company_number=TEST_COMPANY_NUMBER,
            context_pages=1,
        )


@pytest.mark.asyncio
async def test_investigate_multi_year_rejects_an_aggregate_citation_not_drawn_from_any_year(
    session: AsyncSession, company: Company
) -> None:
    """The final aggregation Finding's citations must come from the union of pages actually retrieved across every year's pass, not be invented fresh."""
    extraction_2021 = await _create_filing_with_pages(
        session,
        "investigation-multi-year-badagg-2021",
        ["Willow granite obsidian disclosure for 2021."],
        made_up_date="2021-07-31",
    )
    extraction_2022 = await _create_filing_with_pages(
        session,
        "investigation-multi-year-badagg-2022",
        ["Willow granite obsidian disclosure for 2022."],
        made_up_date="2022-07-31",
    )

    finding_2021 = Finding(
        claim="Willow granite obsidian disclosure for 2021.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2021.id,
                page_number=1,
                supporting_text="disclosure for 2021",
            )
        ],
    )
    finding_2022 = Finding(
        claim="Willow granite obsidian disclosure for 2022.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2022.id,
                page_number=1,
                supporting_text="disclosure for 2022",
            )
        ],
    )
    hallucinated_aggregate = Finding(
        claim="Fabricated comparison citing a page never retrieved in any year.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2021.id,
                page_number=99,
                supporting_text="does not exist",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="willow granite obsidian",
        finding_selector=_finding_selector_by_year(
            {"2021": finding_2021, "2022": finding_2022}, hallucinated_aggregate
        ),
    )

    with pytest.raises(InvestigationAgentError):
        await investigate(
            session,
            chat_client,
            "What did willow granite obsidian disclose from 2021 to 2022?",
            company_number=TEST_COMPANY_NUMBER,
            context_pages=1,
        )


@pytest.mark.asyncio
async def test_investigate_self_corrects_a_fabricated_quote_on_retry(
    session: AsyncSession, company: Company
) -> None:
    """A citation whose supporting_text is not verbatim page text must trigger one retry, not an immediate rejection - the model gets a chance to requote correctly (see README's "Verifying citation quotes" section for the real-run case this is modelled on)."""
    extraction = await _create_filing_with_pages(
        session,
        "investigation-quote-retry-success",
        ["Amber lichen thistle disclosure states the figure was 42 in total."],
    )
    fabricated_finding = Finding(
        claim="Amber lichen thistle figure was 42.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="figure was forty-two exactly",
            )
        ],
    )
    corrected_finding = Finding(
        claim="Amber lichen thistle figure was 42.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="Amber lichen thistle disclosure states the figure was 42 in total",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="amber lichen thistle",
        finding_selector=_finding_selector_with_retry(
            fabricated_finding, corrected_finding
        ),
    )

    finding = await investigate(
        session,
        chat_client,
        "What did amber lichen thistle disclose?",
        company_number=TEST_COMPANY_NUMBER,
    )

    assert finding == corrected_finding
    # 1 fabricated attempt + 1 retry, plus 1 claim_type reclassification call
    # once synthesis succeeds (see `_apply_review_integrity_checks`).
    assert len(chat_client.complete_structured_calls) == 3
    retry_prompt = chat_client.complete_structured_calls[1][-1].content
    assert _RETRY_MARKER in retry_prompt
    assert "figure was forty-two exactly" in retry_prompt


@pytest.mark.asyncio
async def test_investigate_rejects_a_quote_still_fabricated_after_retry(
    session: AsyncSession, company: Company
) -> None:
    """Only one self-correction retry is attempted - a still-fabricated quote after that must raise, not loop indefinitely."""
    extraction = await _create_filing_with_pages(
        session,
        "investigation-quote-retry-failure",
        ["Basil driftwood ember disclosure states the total was 17."],
    )
    fabricated_finding = Finding(
        claim="Basil driftwood ember total was 17.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="the total was definitely seventeen",
            )
        ],
    )
    still_fabricated_finding = Finding(
        claim="Basil driftwood ember total was 17.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="a completely different invented quote",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="basil driftwood ember",
        finding_selector=_finding_selector_with_retry(
            fabricated_finding, still_fabricated_finding
        ),
    )

    with pytest.raises(InvestigationAgentError):
        await investigate(
            session,
            chat_client,
            "What did basil driftwood ember disclose?",
            company_number=TEST_COMPANY_NUMBER,
        )

    assert len(chat_client.complete_structured_calls) == 2


@pytest.mark.asyncio
async def test_investigate_multi_year_self_corrects_a_fabricated_quote_in_a_year_pass(
    session: AsyncSession, company: Company
) -> None:
    """Quote verification and self-correction must also apply inside gather_year_findings_node's per-year passes, not only the single-year path."""
    extraction_2021 = await _create_filing_with_pages(
        session,
        "investigation-quote-retry-multiyear-2021",
        ["Cedar hollow mercury disclosure states the reading was 8 for 2021."],
        made_up_date="2021-07-31",
    )
    extraction_2022 = await _create_filing_with_pages(
        session,
        "investigation-quote-retry-multiyear-2022",
        ["Cedar hollow mercury disclosure states the reading was 9 for 2022."],
        made_up_date="2022-07-31",
    )
    fabricated_2021 = Finding(
        claim="Cedar hollow mercury reading was 8 in 2021.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2021.id,
                page_number=1,
                supporting_text="the reading was eight exactly",
            )
        ],
    )
    corrected_2021 = Finding(
        claim="Cedar hollow mercury reading was 8 in 2021.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2021.id,
                page_number=1,
                supporting_text="Cedar hollow mercury disclosure states the reading was 8 for 2021",
            )
        ],
    )
    finding_2022 = Finding(
        claim="Cedar hollow mercury reading was 9 in 2022.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2022.id,
                page_number=1,
                supporting_text="Cedar hollow mercury disclosure states the reading was 9 for 2022",
            )
        ],
    )
    aggregate_finding = Finding(
        claim="Cedar hollow mercury reading rose from 8 in 2021 to 9 in 2022.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            corrected_2021.citations[0],
            finding_2022.citations[0],
        ],
    )

    def selector(messages: Sequence[ChatMessage]) -> Finding:
        content = messages[-1].content
        if "Per-year findings:" in content:
            return aggregate_finding
        if "fiscal year 2021" in content:
            return corrected_2021 if _RETRY_MARKER in content else fabricated_2021
        if "fiscal year 2022" in content:
            return finding_2022
        raise AssertionError(f"No fake finding configured for prompt: {content!r}")

    chat_client = FakeChatClient(
        query="cedar hollow mercury", finding_selector=selector
    )

    finding = await investigate(
        session,
        chat_client,
        "How did the cedar hollow mercury reading change from 2021 to 2022?",
        company_number=TEST_COMPANY_NUMBER,
        context_pages=1,
    )

    assert finding == aggregate_finding
    # 2021 costs two calls (fabricated + retry), 2022 costs one, aggregation
    # costs one = 4 raw synthesis calls, plus one claim_type reclassification
    # call per successfully-synthesized finding (2021, 2022, aggregate) = 3
    # more (see `_apply_review_integrity_checks`).
    assert len(chat_client.complete_structured_calls) == 7


@pytest.mark.asyncio
async def test_investigate_multi_year_self_corrects_a_fabricated_aggregate_quote(
    session: AsyncSession, company: Company
) -> None:
    """The aggregation call is subject to the same quote verification as the per-year passes, even though it is meant to copy citations verbatim from them."""
    extraction_2021 = await _create_filing_with_pages(
        session,
        "investigation-agg-quote-retry-2021",
        ["Fern quartz lantern disclosure states the count was 5 for 2021."],
        made_up_date="2021-07-31",
    )
    extraction_2022 = await _create_filing_with_pages(
        session,
        "investigation-agg-quote-retry-2022",
        ["Fern quartz lantern disclosure states the count was 6 for 2022."],
        made_up_date="2022-07-31",
    )
    finding_2021 = Finding(
        claim="Fern quartz lantern count was 5 in 2021.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2021.id,
                page_number=1,
                supporting_text="Fern quartz lantern disclosure states the count was 5 for 2021",
            )
        ],
    )
    finding_2022 = Finding(
        claim="Fern quartz lantern count was 6 in 2022.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2022.id,
                page_number=1,
                supporting_text="Fern quartz lantern disclosure states the count was 6 for 2022",
            )
        ],
    )
    fabricated_aggregate = Finding(
        claim="Fern quartz lantern count rose from 5 in 2021 to 6 in 2022.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2021.id,
                page_number=1,
                supporting_text="the count was five, not fabricated at all",
            )
        ],
    )
    corrected_aggregate = Finding(
        claim="Fern quartz lantern count rose from 5 in 2021 to 6 in 2022.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[finding_2021.citations[0], finding_2022.citations[0]],
    )

    def selector(messages: Sequence[ChatMessage]) -> Finding:
        content = messages[-1].content
        if "Per-year findings:" in content:
            return (
                corrected_aggregate
                if _RETRY_MARKER in content
                else fabricated_aggregate
            )
        if "fiscal year 2021" in content:
            return finding_2021
        if "fiscal year 2022" in content:
            return finding_2022
        raise AssertionError(f"No fake finding configured for prompt: {content!r}")

    chat_client = FakeChatClient(query="fern quartz lantern", finding_selector=selector)

    finding = await investigate(
        session,
        chat_client,
        "How did the fern quartz lantern count change from 2021 to 2022?",
        company_number=TEST_COMPANY_NUMBER,
        context_pages=1,
    )

    assert finding == corrected_aggregate
    # 2021 + 2022 cost one call each, aggregation costs two (fabricated +
    # retry) = 4 raw synthesis calls, plus one claim_type reclassification
    # call per successfully-synthesized finding (2021, 2022, aggregate) = 3
    # more (see `_apply_review_integrity_checks`).
    assert len(chat_client.complete_structured_calls) == 7


@pytest.mark.asyncio
async def test_investigate_tolerates_a_clean_quote_against_ocr_noisy_digit_separators(
    session: AsyncSession, company: Company
) -> None:
    """Regression test for a real observed failure: the page's OCR text uses "." instead of "," as a thousands separator and stray underscore leaders (e.g. "437.629 __260.674"), but the model naturally quotes the clean form ("437,629 260,674"). This must pass on the first attempt, not be treated as a fabricated quote (see README's "Verifying citation quotes" section)."""
    extraction = await _create_filing_with_pages(
        session,
        "investigation-ocr-noise-digits",
        ["Hazel current turnover total 437.629 __260.674 for the noted period."],
    )
    clean_quote_finding = Finding(
        claim="Hazel current turnover total was 437,629.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="Hazel current turnover total 437,629 260,674 for the noted period",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="hazel current turnover", finding=clean_quote_finding
    )

    finding = await investigate(
        session,
        chat_client,
        "What was hazel current turnover total?",
        company_number=TEST_COMPANY_NUMBER,
    )

    assert finding == clean_quote_finding
    # 1 synthesis call, plus 1 claim_type reclassification call (see
    # `_apply_review_integrity_checks`).
    assert len(chat_client.complete_structured_calls) == 2


@pytest.mark.asyncio
async def test_investigate_tolerates_a_clean_quote_against_ocr_mismatched_brackets(
    session: AsyncSession, company: Company
) -> None:
    """Regression test for a real observed failure: the page's OCR text pairs a curly brace with a parenthesis (e.g. "{Appointed 9 January 2023)"), but the model naturally quotes matched parentheses. This must pass on the first attempt."""
    extraction = await _create_filing_with_pages(
        session,
        "investigation-ocr-noise-brackets",
        ["Ivory falcon meridian appointed {Appointed 9 January 2023) to the board."],
    )
    clean_quote_finding = Finding(
        claim="Ivory falcon meridian was appointed 9 January 2023.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="Ivory falcon meridian appointed (Appointed 9 January 2023) to the board",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="ivory falcon meridian", finding=clean_quote_finding
    )

    finding = await investigate(
        session,
        chat_client,
        "When was ivory falcon meridian appointed?",
        company_number=TEST_COMPANY_NUMBER,
    )

    assert finding == clean_quote_finding
    # 1 synthesis call, plus 1 claim_type reclassification call (see
    # `_apply_review_integrity_checks`).
    assert len(chat_client.complete_structured_calls) == 2


@pytest.mark.asyncio
async def test_investigate_still_rejects_a_genuinely_different_fabricated_number(
    session: AsyncSession, company: Company
) -> None:
    """The OCR-noise tolerance above must not become so permissive that a genuinely different, fabricated figure slips through undetected."""
    extraction = await _create_filing_with_pages(
        session,
        "investigation-ocr-noise-negative",
        ["Juniper opal cascade total was 100 for the period."],
    )
    fabricated_finding = Finding(
        claim="Juniper opal cascade total was 999.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="total was 999 for the period",
            )
        ],
    )
    chat_client = FakeChatClient(
        query="juniper opal cascade", finding=fabricated_finding
    )

    with pytest.raises(InvestigationAgentError):
        await investigate(
            session,
            chat_client,
            "What was juniper opal cascade total?",
            company_number=TEST_COMPANY_NUMBER,
        )


@pytest.mark.asyncio
async def test_investigate_with_usage_sums_query_and_synthesis_calls(
    session: AsyncSession, company: Company
) -> None:
    await _create_filing_with_pages(
        session,
        "investigation-usage-single-year",
        ["Foxtrot golf hotel disclosure of a figure."],
    )
    finding = Finding(
        claim="Foxtrot golf hotel figure disclosed.",
        claim_type="fact",
        evidence_sufficient=False,
        citations=[],
    )
    per_call_usage = ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chat_client = FakeChatClient(
        query="zqxvwkploqnhfbyt", finding=finding, usage=per_call_usage
    )

    result_finding, usage = await investigate_with_usage(
        session,
        chat_client,
        "What is completely unrelated to this corpus?",
        company_number=TEST_COMPANY_NUMBER,
    )

    assert result_finding == finding
    # One call for generate_query, one for synthesize_finding, one for the
    # claim_type reclassification check (see
    # `_apply_review_integrity_checks`; still runs even though
    # evidence_sufficient is already False, since it only checks claim_type).
    assert usage == ChatUsage(prompt_tokens=30, completion_tokens=15, total_tokens=45)


@pytest.mark.asyncio
async def test_investigate_with_usage_returns_none_when_client_reports_none(
    session: AsyncSession, company: Company
) -> None:
    finding = Finding(
        claim="Unknown.", claim_type="fact", evidence_sufficient=False, citations=[]
    )
    chat_client = FakeChatClient(query="zqxvwkploqnhfbyt", finding=finding)

    _finding, usage = await investigate_with_usage(
        session,
        chat_client,
        "What is completely unrelated to this corpus?",
        company_number=TEST_COMPANY_NUMBER,
    )

    assert usage is None


@pytest.mark.asyncio
async def test_investigate_with_usage_counts_a_self_correction_retry(
    session: AsyncSession, company: Company
) -> None:
    extraction = await _create_filing_with_pages(
        session,
        "investigation-usage-retry",
        ["Hotel india juliet disclosure states the figure was 7 in total."],
    )
    fabricated_finding = Finding(
        claim="Hotel india juliet figure was 7.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="figure was seven exactly",
            )
        ],
    )
    corrected_finding = Finding(
        claim="Hotel india juliet figure was 7.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="Hotel india juliet disclosure states the figure was 7 in total",
            )
        ],
    )
    per_call_usage = ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chat_client = FakeChatClient(
        query="hotel india juliet",
        finding_selector=_finding_selector_with_retry(
            fabricated_finding, corrected_finding
        ),
        usage=per_call_usage,
    )

    result_finding, usage = await investigate_with_usage(
        session,
        chat_client,
        "What did hotel india juliet disclose?",
        company_number=TEST_COMPANY_NUMBER,
    )

    assert result_finding == corrected_finding
    # generate_query + the initial synthesis attempt + the retry + the
    # claim_type reclassification call once synthesis succeeds.
    assert usage == ChatUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60)


@pytest.mark.asyncio
async def test_investigate_with_usage_sums_across_a_multi_year_question(
    session: AsyncSession, company: Company
) -> None:
    extraction_2021 = await _create_filing_with_pages(
        session,
        "investigation-usage-multiyear-2021",
        ["Kilo lima mike figure was 1 for 2021."],
        made_up_date="2021-07-31",
    )
    extraction_2022 = await _create_filing_with_pages(
        session,
        "investigation-usage-multiyear-2022",
        ["Kilo lima mike figure was 2 for 2022."],
        made_up_date="2022-07-31",
    )
    finding_2021 = Finding(
        claim="Kilo lima mike figure was 1 in 2021.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2021.id,
                page_number=1,
                supporting_text="Kilo lima mike figure was 1",
            )
        ],
    )
    finding_2022 = Finding(
        claim="Kilo lima mike figure was 2 in 2022.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction_2022.id,
                page_number=1,
                supporting_text="Kilo lima mike figure was 2",
            )
        ],
    )
    aggregate_finding = Finding(
        claim="Kilo lima mike figure rose from 1 in 2021 to 2 in 2022.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[finding_2021.citations[0], finding_2022.citations[0]],
    )
    per_call_usage = ChatUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chat_client = FakeChatClient(
        query="kilo lima mike",
        finding_selector=_finding_selector_by_year(
            {"2021": finding_2021, "2022": finding_2022}, aggregate_finding
        ),
        usage=per_call_usage,
    )

    result_finding, usage = await investigate_with_usage(
        session,
        chat_client,
        "How did the kilo lima mike figure change from 2021 to 2022?",
        company_number=TEST_COMPANY_NUMBER,
        context_pages=1,
    )

    assert result_finding == aggregate_finding
    # generate_query + one synthesis per year (2) + the final aggregation (4
    # calls), plus one claim_type reclassification call for each of the 3
    # successfully-synthesized findings (2021, 2022, aggregate).
    assert usage == ChatUsage(prompt_tokens=70, completion_tokens=35, total_tokens=105)


@pytest.mark.asyncio
async def test_investigate_with_review_does_not_flag_a_sufficient_fact(
    session: AsyncSession, company: Company
) -> None:
    extraction = await _create_filing_with_pages(
        session,
        "investigation-review-fact",
        ["Romeo sierra tango figure was 100."],
    )
    finding = Finding(
        claim="Romeo sierra tango figure was 100.",
        claim_type="fact",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="figure was 100",
            )
        ],
    )
    chat_client = FakeChatClient(query="romeo sierra tango", finding=finding)

    result_finding, review_id = await investigate_with_review(
        session,
        chat_client,
        "What was the romeo sierra tango figure?",
        company_number=TEST_COMPANY_NUMBER,
    )

    assert result_finding == finding
    assert review_id is None


@pytest.mark.asyncio
async def test_investigate_with_review_flags_an_interpretation_for_review(
    session: AsyncSession, company: Company
) -> None:
    extraction = await _create_filing_with_pages(
        session,
        "investigation-review-interpretation",
        ["Uniform victor whiskey shows three resignations within 14 months."],
    )
    finding = Finding(
        claim="This indicates governance instability.",
        claim_type="interpretation",
        evidence_sufficient=True,
        citations=[
            Citation(
                document_extraction_id=extraction.id,
                page_number=1,
                supporting_text="Uniform victor whiskey shows three resignations within 14 months",
            )
        ],
    )
    chat_client = FakeChatClient(query="uniform victor whiskey", finding=finding)

    result_finding, review_id = await investigate_with_review(
        session,
        chat_client,
        "Does the evidence show governance instability?",
        company_number=TEST_COMPANY_NUMBER,
    )

    assert result_finding == finding
    assert review_id is not None
    persisted = await session.get(HumanReview, review_id)
    assert persisted is not None
    assert persisted.status == "pending"
    assert persisted.claim_type == "interpretation"
    assert persisted.review_reason == "claim_type=interpretation"


@pytest.mark.asyncio
async def test_investigate_with_review_flags_insufficient_evidence_for_review(
    session: AsyncSession, company: Company
) -> None:
    insufficient_finding = Finding(
        claim="The retrieved evidence does not address this question.",
        claim_type="fact",
        evidence_sufficient=False,
        citations=[],
    )
    chat_client = FakeChatClient(query="zqxvwkploqnhfbyt", finding=insufficient_finding)

    _finding, review_id = await investigate_with_review(
        session,
        chat_client,
        "What was the unrelated figure?",
        company_number=TEST_COMPANY_NUMBER,
    )

    assert review_id is not None
    persisted = await session.get(HumanReview, review_id)
    assert persisted is not None
    assert persisted.review_reason == "evidence_sufficient=false"

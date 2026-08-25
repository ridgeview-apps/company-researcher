from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
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
)
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.entailment_judge import EntailmentJudgment
from company_researcher.judge_calibration import (
    EntailmentExample,
    ExampleResult,
    JudgeCalibrationError,
    _resolve_page_text,
    _summarize,
    evaluate_example,
    load_entailment_dataset,
    run_calibration,
)
from company_researcher.llm_client import ChatMessage

_StructuredResponse = TypeVar("_StructuredResponse", bound=BaseModel)

TEST_COMPANY_NUMBER = "TE000011"
REPO_ROOT = Path(__file__).resolve().parent.parent
GYMSHARK_DATASET_PATH = REPO_ROOT / "evaluation" / "citation_entailment_judgments.json"
GYMSHARK_COMPANY_NUMBER = "08130873"


class FakeCalibrationChatClient:
    """Returns a judgment chosen per-example via a selector, or a fixed one."""

    def __init__(
        self,
        *,
        judgment: EntailmentJudgment | None = None,
        judgment_selector: Callable[[str], EntailmentJudgment] | None = None,
    ) -> None:
        self._judgment = judgment
        self._judgment_selector = judgment_selector

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        raise AssertionError("calibration should not call complete()")

    async def complete_structured(
        self, messages: Sequence[ChatMessage], response_model: type[_StructuredResponse]
    ) -> _StructuredResponse:
        content = messages[-1].content
        if self._judgment_selector is not None:
            return cast(_StructuredResponse, self._judgment_selector(content))
        assert self._judgment is not None
        return cast(_StructuredResponse, self._judgment)


def test_load_entailment_dataset_parses_the_gymshark_fixture() -> None:
    dataset = load_entailment_dataset(GYMSHARK_DATASET_PATH)

    assert dataset.company_number == GYMSHARK_COMPANY_NUMBER
    assert len(dataset.examples) == 14
    first = dataset.examples[0]
    assert first.id == "e01-fy2023-turnover-total-supported"
    assert first.human_verdict == "supported"


def test_summarize_computes_accuracy_precision_recall_f1() -> None:
    results = (
        ExampleResult(
            example_id="tp",
            human_verdict="unsupported",
            judge_verdict="unsupported",
            judge_reason="",
        ),
        ExampleResult(
            example_id="fp",
            human_verdict="supported",
            judge_verdict="unsupported",
            judge_reason="",
        ),
        ExampleResult(
            example_id="fn",
            human_verdict="unsupported",
            judge_verdict="supported",
            judge_reason="",
        ),
        ExampleResult(
            example_id="tn",
            human_verdict="supported",
            judge_verdict="supported",
            judge_reason="",
        ),
    )

    summary = _summarize(results)

    assert summary.accuracy == pytest.approx(0.5)
    assert summary.precision_unsupported == pytest.approx(0.5)
    assert summary.recall_unsupported == pytest.approx(0.5)
    assert summary.f1_unsupported == pytest.approx(0.5)


def test_summarize_handles_no_predicted_unsupported_without_dividing_by_zero() -> None:
    results = (
        ExampleResult(
            example_id="tn1",
            human_verdict="supported",
            judge_verdict="supported",
            judge_reason="",
        ),
        ExampleResult(
            example_id="fn1",
            human_verdict="unsupported",
            judge_verdict="supported",
            judge_reason="",
        ),
    )

    summary = _summarize(results)

    assert summary.precision_unsupported == 0.0
    assert summary.recall_unsupported == 0.0
    assert summary.f1_unsupported == 0.0


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
        company_name="JUDGE CALIBRATION TEST LIMITED",
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
async def test_resolve_page_text_returns_the_real_persisted_text(
    session: AsyncSession, company: Company
) -> None:
    await _create_filing_with_pages(
        session, "calibration-transaction-alpha", ["Alpha page one.", "Alpha page two."]
    )

    text = await _resolve_page_text(
        session, TEST_COMPANY_NUMBER, "calibration-transaction-alpha", 2
    )

    assert text == "Alpha page two."


@pytest.mark.asyncio
async def test_resolve_page_text_raises_for_a_missing_page(
    session: AsyncSession, company: Company
) -> None:
    await _create_filing_with_pages(
        session, "calibration-transaction-beta", ["Only page."]
    )

    with pytest.raises(JudgeCalibrationError):
        await _resolve_page_text(
            session, TEST_COMPANY_NUMBER, "calibration-transaction-beta", 99
        )


@pytest.mark.asyncio
async def test_evaluate_example_compares_judge_verdict_to_human_label(
    session: AsyncSession, company: Company
) -> None:
    await _create_filing_with_pages(
        session,
        "calibration-transaction-gamma",
        ["Turnover was 100 in the class-of-business note."],
    )
    example = EntailmentExample(
        id="gamma-1",
        transaction_id="calibration-transaction-gamma",
        page_number=1,
        claim="Turnover was 100.",
        supporting_text="Turnover was 100",
        human_verdict="supported",
        human_reason="Matches the page.",
    )
    chat_client = FakeCalibrationChatClient(
        judgment=EntailmentJudgment(verdict="supported", reason="Matches.")
    )

    result = await evaluate_example(session, chat_client, example, TEST_COMPANY_NUMBER)

    assert result.example_id == "gamma-1"
    assert result.human_verdict == "supported"
    assert result.judge_verdict == "supported"
    assert result.agrees


@pytest.mark.asyncio
async def test_run_calibration_resolves_the_real_gymshark_fixture(
    session: AsyncSession,
) -> None:
    """Every example's transaction_id/page_number resolves against the real persisted corpus.

    Uses a fake chat client (always agreeing with the human label) so this
    test proves dataset resolution against real data without depending on
    a real LLM call - the same split this project's other evaluation
    fixture tests already use.
    """
    dataset = load_entailment_dataset(GYMSHARK_DATASET_PATH)

    def agree_with_human(content: str) -> EntailmentJudgment:
        for example in dataset.examples:
            if example.claim in content:
                return EntailmentJudgment(
                    verdict=example.human_verdict, reason="fake agreement"
                )
        raise AssertionError(f"No example matched prompt: {content!r}")

    chat_client = FakeCalibrationChatClient(judgment_selector=agree_with_human)

    summary = await run_calibration(session, chat_client, dataset)

    assert len(summary.per_example) == len(dataset.examples)
    assert summary.accuracy == pytest.approx(1.0)

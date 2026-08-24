from collections.abc import AsyncIterator, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.config import Settings
from company_researcher.db.models import (
    EMBEDDING_DIMENSIONS,
    Company,
    DocumentExtraction,
    DocumentPage,
    Filing,
    FilingDocument,
)
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.embedding_persistence import embed_document_extraction
from company_researcher.retrieval_evaluation import (
    EvaluationQuestion,
    RelevantPage,
    RetrievalEvaluationError,
    evaluate_question,
    evaluate_question_by_embedding,
    load_evaluation_dataset,
    run_evaluation,
)

FAKE_EMBEDDING_PROVIDER = "fake"
FAKE_EMBEDDING_MODEL = "fake-model"


def _axis_vector(index: int, value: float = 1.0) -> list[float]:
    """A vector with `value` at `index` and zero elsewhere, for clean cosine geometry."""
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[index] = value
    return vector


class FakeEmbeddingsProvider:
    """Embeds each text by exact lookup, for deterministic vector-search tests."""

    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vectors_by_text[text] for text in texts]


TEST_COMPANY_NUMBER = "TE000007"
REPO_ROOT = Path(__file__).resolve().parent.parent
GYMSHARK_DATASET_PATH = REPO_ROOT / "evaluation" / "gymshark_retrieval_questions.json"
GYMSHARK_COMPANY_NUMBER = "08130873"


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


@pytest_asyncio.fixture
async def company(session: AsyncSession) -> Company:
    company = Company(
        company_number=TEST_COMPANY_NUMBER,
        company_name="RETRIEVAL EVALUATION TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=datetime.now(UTC),
    )
    session.add(company)
    await session.commit()
    return company


@pytest.mark.asyncio
async def test_evaluate_question_scores_a_retrieved_relevant_page(
    session: AsyncSession, company: Company
) -> None:
    await _create_filing_with_pages(
        session,
        "eval-transaction-alpha",
        ["Alpha bravo charlie appears on this page.", "Delta echo foxtrot."],
    )
    question = EvaluationQuestion(
        id="q-alpha",
        text="Where does alpha bravo appear?",
        query="alpha bravo",
        relevant_pages=(
            RelevantPage(transaction_id="eval-transaction-alpha", page_number=1),
        ),
    )

    metrics = await evaluate_question(
        session, question, TEST_COMPANY_NUMBER, k_values=(1, 3), search_depth=10
    )

    assert metrics.recall_at_k == {1: 1.0, 3: 1.0}
    assert metrics.reciprocal_rank == 1.0


@pytest.mark.asyncio
async def test_evaluate_question_scores_zero_for_unretrieved_page(
    session: AsyncSession, company: Company
) -> None:
    await _create_filing_with_pages(
        session,
        "eval-transaction-beta",
        ["Golf hotel india on this page."],
    )
    question = EvaluationQuestion(
        id="q-beta",
        text="Where does zulu yankee whiskey appear?",
        query="zulu yankee whiskey",
        relevant_pages=(
            RelevantPage(transaction_id="eval-transaction-beta", page_number=1),
        ),
    )

    metrics = await evaluate_question(
        session, question, TEST_COMPANY_NUMBER, k_values=(1,), search_depth=10
    )

    assert metrics.recall_at_k == {1: 0.0}
    assert metrics.reciprocal_rank == 0.0


@pytest.mark.asyncio
async def test_evaluate_question_raises_for_unresolvable_transaction_id(
    session: AsyncSession, company: Company
) -> None:
    question = EvaluationQuestion(
        id="q-missing",
        text="Anything?",
        query="anything",
        relevant_pages=(RelevantPage(transaction_id="does-not-exist", page_number=1),),
    )

    with pytest.raises(RetrievalEvaluationError):
        await evaluate_question(session, question, TEST_COMPANY_NUMBER)


@pytest.mark.asyncio
async def test_evaluate_question_by_embedding_scores_a_retrieved_relevant_page(
    session: AsyncSession, company: Company
) -> None:
    first_page_text = "Alpha bravo charlie appears on this page."
    second_page_text = "Delta echo foxtrot."
    question_text = "Where does alpha bravo appear?"
    extraction = await _create_filing_with_pages(
        session, "eval-vector-transaction-alpha", [first_page_text, second_page_text]
    )
    provider = FakeEmbeddingsProvider(
        {
            first_page_text: _axis_vector(0),
            second_page_text: _axis_vector(1),
            question_text: _axis_vector(0),
        }
    )
    await embed_document_extraction(
        session,
        provider,
        extraction,
        provider=FAKE_EMBEDDING_PROVIDER,
        model=FAKE_EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    question = EvaluationQuestion(
        id="q-alpha-vector",
        text=question_text,
        query="unused",
        relevant_pages=(
            RelevantPage(transaction_id="eval-vector-transaction-alpha", page_number=1),
        ),
    )

    metrics = await evaluate_question_by_embedding(
        session,
        provider,
        question,
        TEST_COMPANY_NUMBER,
        provider=FAKE_EMBEDDING_PROVIDER,
        model=FAKE_EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        k_values=(1, 3),
        search_depth=10,
    )

    assert metrics.recall_at_k == {1: 1.0, 3: 1.0}
    assert metrics.reciprocal_rank == 1.0


@pytest.mark.asyncio
async def test_evaluate_question_by_embedding_scores_zero_beyond_search_depth(
    session: AsyncSession, company: Company
) -> None:
    """Unlike lexical search's match/no-match threshold, vector search always
    returns *something* up to `search_depth` — a relevant page only scores
    zero if it ranks beyond that limit, not because nothing "matched"."""
    relevant_page_text = "Golf hotel india on this page."
    closer_page_text = "Unrelated content that happens to rank closer."
    question_text = "Where does golf hotel india appear?"
    extraction = await _create_filing_with_pages(
        session,
        "eval-vector-transaction-beta",
        [relevant_page_text, closer_page_text],
    )
    provider = FakeEmbeddingsProvider(
        {
            relevant_page_text: _axis_vector(1),  # far from the question vector
            closer_page_text: _axis_vector(0),  # identical to the question vector
            question_text: _axis_vector(0),
        }
    )
    await embed_document_extraction(
        session,
        provider,
        extraction,
        provider=FAKE_EMBEDDING_PROVIDER,
        model=FAKE_EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    question = EvaluationQuestion(
        id="q-beta-vector",
        text=question_text,
        query="unused",
        relevant_pages=(
            RelevantPage(transaction_id="eval-vector-transaction-beta", page_number=1),
        ),
    )

    metrics = await evaluate_question_by_embedding(
        session,
        provider,
        question,
        TEST_COMPANY_NUMBER,
        provider=FAKE_EMBEDDING_PROVIDER,
        model=FAKE_EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        k_values=(1,),
        search_depth=1,
    )

    assert metrics.recall_at_k == {1: 0.0}
    assert metrics.reciprocal_rank == 0.0


def test_load_evaluation_dataset_parses_gymshark_fixture() -> None:
    dataset = load_evaluation_dataset(GYMSHARK_DATASET_PATH)

    assert dataset.company_number == GYMSHARK_COMPANY_NUMBER
    assert len(dataset.questions) == 6
    assert all(question.relevant_pages for question in dataset.questions)


@pytest.mark.asyncio
async def test_run_evaluation_resolves_the_gymshark_fixture_against_real_data() -> None:
    """Guard against the labelled dataset drifting from the persisted corpus."""
    dataset = load_evaluation_dataset(GYMSHARK_DATASET_PATH)
    engine = create_database_engine(Settings(_env_file=None))  # type: ignore[call-arg]
    session_factory = create_session_factory(engine)
    try:
        async with session_factory() as session:
            summary = await run_evaluation(session, dataset)
    finally:
        await engine.dispose()

    assert len(summary.per_question) == len(dataset.questions)
    for metrics in summary.per_question:
        for recall in metrics.recall_at_k.values():
            assert 0.0 <= recall <= 1.0
        assert 0.0 <= metrics.reciprocal_rank <= 1.0

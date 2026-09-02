import json
from dataclasses import dataclass, replace
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import DocumentExtraction, Filing, FilingDocument
from company_researcher.discriminative_query import derive_discriminative_query
from company_researcher.embeddings_client import EmbeddingsProvider
from company_researcher.hybrid_search import reciprocal_rank_fusion
from company_researcher.lexical_search import search_pages
from company_researcher.query_construction import derive_query
from company_researcher.vector_search import search_pages_by_embedding

DEFAULT_K_VALUES: tuple[int, ...] = (5, 10)
DEFAULT_SEARCH_DEPTH = 50


class RetrievalEvaluationError(Exception):
    """Raised when an evaluation dataset cannot be resolved against persisted data."""


@dataclass(frozen=True)
class RelevantPage:
    """A manually labelled relevant page, keyed by stable filing transaction ID."""

    transaction_id: str
    page_number: int


@dataclass(frozen=True)
class EvaluationQuestion:
    """One retrieval question and the pages a human identified as relevant to it.

    `text` is the natural-language question, kept for reporting. `query` is
    the short keyword string actually issued to lexical search: matching the
    full question sentence performs far worse than a short query against
    PostgreSQL's OR-combined term ranking, so the two are kept distinct.
    `note` is the dataset's hand-verified ground-truth answer - unused by
    retrieval scoring itself, but read by `accuracy_scoring.py` to build a
    factual-accuracy review; it defaults to an empty string so existing ad
    hoc test fixtures that construct `EvaluationQuestion` directly don't
    need to supply one.
    """

    id: str
    text: str
    query: str
    relevant_pages: tuple[RelevantPage, ...]
    note: str = ""


@dataclass(frozen=True)
class EvaluationDataset:
    """A labelled retrieval evaluation corpus for one company."""

    company_number: str
    company_name: str
    questions: tuple[EvaluationQuestion, ...]


def load_evaluation_dataset(path: Path) -> EvaluationDataset:
    """Parse a labelled retrieval evaluation dataset from JSON."""
    payload = json.loads(path.read_text())
    questions = tuple(
        EvaluationQuestion(
            id=question["id"],
            text=question["text"],
            query=question["query"],
            relevant_pages=tuple(
                RelevantPage(
                    transaction_id=page["transaction_id"],
                    page_number=page["page_number"],
                )
                for page in question["relevant_pages"]
            ),
            note=question["note"],
        )
        for question in payload["questions"]
    )
    return EvaluationDataset(
        company_number=payload["company_number"],
        company_name=payload["company_name"],
        questions=questions,
    )


def with_derived_queries(dataset: EvaluationDataset) -> EvaluationDataset:
    """Replace each question's hand-picked query with `derive_query(text)`.

    Unlike the dataset's own `query` field, a derived query cannot have been
    tuned to that question's known-relevant pages, since `derive_query`
    depends only on `text`. This gives an honest (if not necessarily better)
    measurement of query wording without hand-tuning bias.
    """
    return EvaluationDataset(
        company_number=dataset.company_number,
        company_name=dataset.company_name,
        questions=tuple(
            replace(question, query=derive_query(question.text))
            for question in dataset.questions
        ),
    )


async def with_discriminative_queries(
    session: AsyncSession, dataset: EvaluationDataset
) -> EvaluationDataset:
    """Replace each question's query with `derive_discriminative_query(session, text)`.

    Like `with_derived_queries`, this cannot leak a specific answer: the
    query depends only on `text` and corpus-wide document-frequency
    statistics, never on which page is known to be relevant.
    """
    questions = tuple(
        [
            replace(
                question,
                query=await derive_discriminative_query(session, question.text),
            )
            for question in dataset.questions
        ]
    )
    return EvaluationDataset(
        company_number=dataset.company_number,
        company_name=dataset.company_name,
        questions=questions,
    )


@dataclass(frozen=True)
class QuestionMetrics:
    """Recall@K and reciprocal rank for one evaluated question."""

    question_id: str
    recall_at_k: dict[int, float]
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Per-question metrics together with corpus-wide means."""

    per_question: tuple[QuestionMetrics, ...]
    mean_recall_at_k: dict[int, float]
    mean_reciprocal_rank: float


async def _resolve_relevant_extraction_pages(
    session: AsyncSession,
    company_number: str,
    relevant_pages: tuple[RelevantPage, ...],
) -> set[tuple[int, int]]:
    """Resolve transaction-ID relevance labels to (document_extraction_id, page_number)."""
    transaction_ids = {page.transaction_id for page in relevant_pages}
    statement = (
        select(Filing.transaction_id, DocumentExtraction.id)
        .join(FilingDocument, FilingDocument.filing_id == Filing.id)
        .join(
            DocumentExtraction,
            DocumentExtraction.filing_document_id == FilingDocument.id,
        )
        .where(
            Filing.company_number == company_number,
            Filing.transaction_id.in_(transaction_ids),
            DocumentExtraction.status == "succeeded",
        )
    )
    result = await session.execute(statement)
    extraction_id_by_transaction = {row.transaction_id: row.id for row in result}

    missing = transaction_ids - extraction_id_by_transaction.keys()
    if missing:
        raise RetrievalEvaluationError(
            "No successful extraction is persisted for filing transaction ID(s): "
            f"{sorted(missing)}"
        )

    return {
        (extraction_id_by_transaction[page.transaction_id], page.page_number)
        for page in relevant_pages
    }


def _score_retrieved_pages(
    question_id: str,
    retrieved: list[tuple[int, int]],
    relevant: set[tuple[int, int]],
    k_values: tuple[int, ...],
) -> QuestionMetrics:
    """Score a ranked list of retrieved pages against known-relevant pages."""
    recall_at_k = {
        k: len(relevant & set(retrieved[:k])) / len(relevant) for k in k_values
    }
    reciprocal_rank = 0.0
    for position, page in enumerate(retrieved, start=1):
        if page in relevant:
            reciprocal_rank = 1 / position
            break

    return QuestionMetrics(
        question_id=question_id,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
    )


def _summarize(
    per_question: tuple[QuestionMetrics, ...], k_values: tuple[int, ...]
) -> EvaluationSummary:
    """Average per-question metrics into corpus-wide Recall@K and MRR."""
    question_count = len(per_question)
    mean_recall_at_k = {
        k: sum(metrics.recall_at_k[k] for metrics in per_question) / question_count
        for k in k_values
    }
    mean_reciprocal_rank = (
        sum(metrics.reciprocal_rank for metrics in per_question) / question_count
    )
    return EvaluationSummary(
        per_question=per_question,
        mean_recall_at_k=mean_recall_at_k,
        mean_reciprocal_rank=mean_reciprocal_rank,
    )


async def evaluate_question(
    session: AsyncSession,
    question: EvaluationQuestion,
    company_number: str,
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
) -> QuestionMetrics:
    """Score one question's lexical search results against its relevance labels."""
    relevant = await _resolve_relevant_extraction_pages(
        session, company_number, question.relevant_pages
    )
    matches = await search_pages(
        session, question.query, limit=search_depth, company_number=company_number
    )
    retrieved = [(match.document_extraction_id, match.page_number) for match in matches]
    return _score_retrieved_pages(question.id, retrieved, relevant, k_values)


async def run_evaluation(
    session: AsyncSession,
    dataset: EvaluationDataset,
    *,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
) -> EvaluationSummary:
    """Evaluate every question in a dataset and summarize Recall@K and MRR."""
    per_question = tuple(
        [
            await evaluate_question(
                session,
                question,
                dataset.company_number,
                k_values=k_values,
                search_depth=search_depth,
            )
            for question in dataset.questions
        ]
    )
    return _summarize(per_question, k_values)


async def evaluate_question_by_embedding(
    session: AsyncSession,
    embeddings_client: EmbeddingsProvider,
    question: EvaluationQuestion,
    company_number: str,
    *,
    provider: str,
    model: str,
    dimensions: int,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
) -> QuestionMetrics:
    """Score one question's vector search results against its relevance labels.

    Embeds the question's full natural-language `text`, not a short keyword
    query: unlike PostgreSQL's OR-combined lexical ranking, embeddings are
    not diluted by extra context, so there is no reason to shorten it here.
    """
    relevant = await _resolve_relevant_extraction_pages(
        session, company_number, question.relevant_pages
    )
    (query_embedding,) = await embeddings_client.embed([question.text])
    matches = await search_pages_by_embedding(
        session,
        query_embedding,
        provider=provider,
        model=model,
        dimensions=dimensions,
        limit=search_depth,
        company_number=company_number,
    )
    retrieved = [(match.document_extraction_id, match.page_number) for match in matches]
    return _score_retrieved_pages(question.id, retrieved, relevant, k_values)


async def run_vector_evaluation(
    session: AsyncSession,
    embeddings_client: EmbeddingsProvider,
    dataset: EvaluationDataset,
    *,
    provider: str,
    model: str,
    dimensions: int,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
) -> EvaluationSummary:
    """Evaluate every question in a dataset using vector search."""
    per_question = tuple(
        [
            await evaluate_question_by_embedding(
                session,
                embeddings_client,
                question,
                dataset.company_number,
                provider=provider,
                model=model,
                dimensions=dimensions,
                k_values=k_values,
                search_depth=search_depth,
            )
            for question in dataset.questions
        ]
    )
    return _summarize(per_question, k_values)


async def evaluate_question_hybrid(
    session: AsyncSession,
    embeddings_client: EmbeddingsProvider,
    question: EvaluationQuestion,
    company_number: str,
    *,
    provider: str,
    model: str,
    dimensions: int,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
) -> QuestionMetrics:
    """Score one question's Reciprocal-Rank-Fused lexical+vector results.

    Reuses each method's own established query input unchanged: `question.query`
    for lexical search, `question.text` embedded fresh for vector search. Both
    rankings are computed to `search_depth` before fusion, since a page can
    only contribute to the fused ranking if it was retrieved in the first
    place.
    """
    relevant = await _resolve_relevant_extraction_pages(
        session, company_number, question.relevant_pages
    )
    lexical_matches = await search_pages(
        session, question.query, limit=search_depth, company_number=company_number
    )
    (query_embedding,) = await embeddings_client.embed([question.text])
    vector_matches = await search_pages_by_embedding(
        session,
        query_embedding,
        provider=provider,
        model=model,
        dimensions=dimensions,
        limit=search_depth,
        company_number=company_number,
    )
    fused = reciprocal_rank_fusion(lexical_matches, vector_matches)
    retrieved = [(match.document_extraction_id, match.page_number) for match in fused]
    return _score_retrieved_pages(question.id, retrieved, relevant, k_values)


async def run_hybrid_evaluation(
    session: AsyncSession,
    embeddings_client: EmbeddingsProvider,
    dataset: EvaluationDataset,
    *,
    provider: str,
    model: str,
    dimensions: int,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    search_depth: int = DEFAULT_SEARCH_DEPTH,
) -> EvaluationSummary:
    """Evaluate every question in a dataset using Reciprocal Rank Fusion."""
    per_question = tuple(
        [
            await evaluate_question_hybrid(
                session,
                embeddings_client,
                question,
                dataset.company_number,
                provider=provider,
                model=model,
                dimensions=dimensions,
                k_values=k_values,
                search_depth=search_depth,
            )
            for question in dataset.questions
        ]
    )
    return _summarize(per_question, k_values)

import json
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import DocumentExtraction, Filing, FilingDocument
from company_researcher.lexical_search import search_pages

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
    """

    id: str
    text: str
    query: str
    relevant_pages: tuple[RelevantPage, ...]


@dataclass(frozen=True)
class EvaluationDataset:
    """A labelled retrieval evaluation corpus for one company."""

    company_number: str
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
        )
        for question in payload["questions"]
    )
    return EvaluationDataset(
        company_number=payload["company_number"], questions=questions
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
    matches = await search_pages(session, question.query, limit=search_depth)
    retrieved = [(match.document_extraction_id, match.page_number) for match in matches]

    recall_at_k = {
        k: len(relevant & set(retrieved[:k])) / len(relevant) for k in k_values
    }
    reciprocal_rank = 0.0
    for position, page in enumerate(retrieved, start=1):
        if page in relevant:
            reciprocal_rank = 1 / position
            break

    return QuestionMetrics(
        question_id=question.id,
        recall_at_k=recall_at_k,
        reciprocal_rank=reciprocal_rank,
    )


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

from pathlib import Path

import pytest

from company_researcher.accuracy_scoring import (
    AccuracyScoringError,
    CitationRef,
    QuestionAccuracyReview,
    generate_accuracy_review,
    load_accuracy_review,
    save_accuracy_review,
    score_accuracy_review,
)
from company_researcher.baseline_comparison import QuestionComparison
from company_researcher.investigation_agent import Citation, Finding
from company_researcher.retrieval_evaluation import (
    EvaluationDataset,
    EvaluationQuestion,
)


def _finding(claim: str, citations: list[Citation] | None = None) -> Finding:
    return Finding(
        claim=claim,
        claim_type="fact",
        evidence_sufficient=True,
        citations=citations or [],
    )


def _comparison(
    question_id: str,
    question_text: str,
    baseline_claim: str,
    *,
    baseline_citations: list[Citation] | None = None,
    specialized_claim: str | None,
    specialized_citations: list[Citation] | None = None,
    specialized_error: str | None = None,
) -> QuestionComparison:
    return QuestionComparison(
        question_id=question_id,
        question_text=question_text,
        baseline_finding=_finding(baseline_claim, baseline_citations),
        baseline_usage=None,
        baseline_latency_seconds=0.1,
        baseline_citation_realism=(),
        specialized_finding=(
            _finding(specialized_claim, specialized_citations)
            if specialized_claim is not None
            else None
        ),
        specialized_usage=None,
        specialized_error=specialized_error,
        specialized_latency_seconds=0.2,
    )


def _dataset(notes: dict[str, str]) -> EvaluationDataset:
    return EvaluationDataset(
        company_number="TE000011",
        company_name="Test Co",
        questions=tuple(
            EvaluationQuestion(
                id=qid, text=qid, query=qid, relevant_pages=(), note=note
            )
            for qid, note in notes.items()
        ),
    )


def test_generate_accuracy_review_builds_a_blank_template_from_real_comparisons() -> (
    None
):
    comparisons = [
        _comparison(
            "q1",
            "What was turnover?",
            "Turnover was £490m",
            baseline_citations=[
                Citation(
                    document_extraction_id=42,
                    page_number=20,
                    supporting_text="Turnover 490,142",
                )
            ],
            specialized_claim="Turnover was £490m",
            specialized_citations=[
                Citation(
                    document_extraction_id=42,
                    page_number=20,
                    supporting_text="Turnover 490,142",
                )
            ],
        ),
        _comparison(
            "q2",
            "Who is the secretary?",
            "I don't know",
            specialized_claim=None,
            specialized_error="cited an unretrieved page",
        ),
    ]
    dataset = _dataset({"q1": "Turnover: £490,142k", "q2": "Secretary: C Reed"})

    reviews = generate_accuracy_review(comparisons, dataset)

    assert reviews[0] == QuestionAccuracyReview(
        question_id="q1",
        question_text="What was turnover?",
        ground_truth_note="Turnover: £490,142k",
        baseline_claim="Turnover was £490m",
        baseline_citations=(
            CitationRef(
                document_extraction_id=42,
                page_number=20,
                supporting_text="Turnover 490,142",
            ),
        ),
        baseline_verdict=None,
        specialized_claim="Turnover was £490m",
        specialized_citations=(
            CitationRef(
                document_extraction_id=42,
                page_number=20,
                supporting_text="Turnover 490,142",
            ),
        ),
        specialized_error=None,
        specialized_verdict=None,
        specialized_refusal_verdict=None,
    )
    assert reviews[1] == QuestionAccuracyReview(
        question_id="q2",
        question_text="Who is the secretary?",
        ground_truth_note="Secretary: C Reed",
        baseline_claim="I don't know",
        baseline_citations=(),
        baseline_verdict=None,
        specialized_claim=None,
        specialized_citations=(),
        specialized_error="cited an unretrieved page",
        specialized_verdict=None,
        specialized_refusal_verdict=None,
    )


def test_generate_accuracy_review_rejects_a_question_missing_from_the_dataset() -> None:
    comparisons = [_comparison("q1", "?", "claim", specialized_claim="claim")]
    dataset = _dataset({"other-question": "note"})

    with pytest.raises(AccuracyScoringError, match="q1"):
        generate_accuracy_review(comparisons, dataset)


def test_save_and_load_accuracy_review_round_trips(tmp_path: Path) -> None:
    reviews = [
        QuestionAccuracyReview(
            question_id="q1",
            question_text="?",
            ground_truth_note="note",
            baseline_claim="baseline claim",
            baseline_citations=(
                CitationRef(
                    document_extraction_id=1, page_number=2, supporting_text="text"
                ),
            ),
            baseline_verdict="correct",
            specialized_claim="specialized claim",
            specialized_citations=(),
            specialized_error=None,
            specialized_verdict="partially_correct",
            specialized_refusal_verdict=None,
        ),
    ]
    path = tmp_path / "review.json"

    save_accuracy_review(reviews, path)
    loaded = load_accuracy_review(path)

    assert loaded == reviews


def _reviewed(
    question_id: str,
    *,
    baseline_verdict: str | None,
    specialized_claim: str | None,
    specialized_verdict: str | None = None,
    specialized_refusal_verdict: str | None = None,
) -> QuestionAccuracyReview:
    return QuestionAccuracyReview(
        question_id=question_id,
        question_text="?",
        ground_truth_note="note",
        baseline_claim="c",
        baseline_citations=(),
        baseline_verdict=baseline_verdict,  # type: ignore[arg-type]
        specialized_claim=specialized_claim,
        specialized_citations=(),
        specialized_error=None if specialized_claim else "refused",
        specialized_verdict=specialized_verdict,  # type: ignore[arg-type]
        specialized_refusal_verdict=specialized_refusal_verdict,  # type: ignore[arg-type]
    )


def test_score_accuracy_review_counts_verdicts_and_refusals_separately() -> None:
    reviews = [
        _reviewed(
            "q1",
            baseline_verdict="correct",
            specialized_claim="c",
            specialized_verdict="correct",
        ),
        _reviewed(
            "q2",
            baseline_verdict="incorrect",
            specialized_claim="c",
            specialized_verdict="partially_correct",
        ),
        _reviewed(
            "q3",
            baseline_verdict="incorrect",
            specialized_claim=None,
            specialized_refusal_verdict="appropriate",
        ),
        _reviewed(
            "q4",
            baseline_verdict="correct",
            specialized_claim=None,
            specialized_refusal_verdict="inappropriate",
        ),
    ]

    report = score_accuracy_review(reviews)

    assert report.baseline.correct == 2
    assert report.baseline.incorrect == 2
    assert report.baseline.total == 4
    assert report.specialized_claims.correct == 1
    assert report.specialized_claims.partially_correct == 1
    assert report.specialized_claims.total == 2
    assert report.specialized_refusals.appropriate == 1
    assert report.specialized_refusals.inappropriate == 1
    assert report.specialized_refusals.total == 2


def test_score_accuracy_review_fails_closed_on_a_missing_baseline_verdict() -> None:
    reviews = [
        _reviewed(
            "q1",
            baseline_verdict=None,
            specialized_claim="c",
            specialized_verdict="correct",
        ),
    ]

    with pytest.raises(AccuracyScoringError, match="q1"):
        score_accuracy_review(reviews)


def test_score_accuracy_review_fails_closed_on_an_inconsistent_refusal_state() -> None:
    reviews = [
        _reviewed(
            "q1",
            baseline_verdict="correct",
            specialized_claim=None,
            specialized_refusal_verdict=None,
        ),
    ]

    with pytest.raises(AccuracyScoringError, match="specialized_refusal_verdict"):
        score_accuracy_review(reviews)


def test_score_accuracy_review_rejects_an_empty_review() -> None:
    with pytest.raises(AccuracyScoringError, match="empty"):
        score_accuracy_review([])

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from company_researcher.baseline_comparison import QuestionComparison
from company_researcher.investigation_agent import Citation
from company_researcher.retrieval_evaluation import EvaluationDataset

AccuracyVerdict = Literal["correct", "partially_correct", "incorrect"]
RefusalVerdict = Literal["appropriate", "inappropriate"]


class AccuracyScoringError(Exception):
    """Raised when an accuracy review cannot be generated or scored."""


@dataclass(frozen=True)
class CitationRef:
    """A claim's citation, kept so a human reviewer can look up the real page text.

    Without this, a review template would leave a reader no way to trace a
    claim back to source - exactly the gap that motivated adding this
    field: the first version of this schema only kept claim text, and a
    reviewer trying to verify a claim against the real corpus had nothing
    to query by.
    """

    document_extraction_id: int
    page_number: int
    supporting_text: str


def _citation_ref(citation: Citation) -> CitationRef:
    return CitationRef(
        document_extraction_id=citation.document_extraction_id,
        page_number=citation.page_number,
        supporting_text=citation.supporting_text,
    )


@dataclass(frozen=True)
class QuestionAccuracyReview:
    """One question's ground truth and both baselines' real claims, for human review.

    The no-retrieval baseline always produces a `claim` (its structured
    output always includes one, even when it self-reports
    `evidence_sufficient=False` - see README's "Compare the specialized
    agent against a general-LLM baseline" section), so `baseline_verdict`
    is always the 3-way correctness scale. The specialized agent sometimes
    produces no claim at all, raising `InvestigationAgentError` and
    refusing rather than guess - refusing is not itself a wrong answer, so
    it is judged on a separate axis (`specialized_refusal_verdict`: was
    refusing the right call given the evidence actually available) rather
    than forced into the same correctness scale, which would conflate
    "answered incorrectly" with "declined to answer." Exactly one of
    `specialized_verdict`/`specialized_refusal_verdict` applies to a given
    question, matching whether `specialized_claim` is present or `None`.

    Verdict fields are `None` when generated from a real comparison run -
    a human must fill them in by comparing each claim against
    `ground_truth_note` (and, via each claim's citations, the real
    persisted page text) before this can be scored; see
    `score_accuracy_review`, which fails closed on any question still
    unreviewed rather than silently excluding it.
    """

    question_id: str
    question_text: str
    ground_truth_note: str
    baseline_claim: str
    baseline_citations: tuple[CitationRef, ...]
    baseline_verdict: AccuracyVerdict | None
    specialized_claim: str | None
    specialized_citations: tuple[CitationRef, ...]
    specialized_error: str | None
    specialized_verdict: AccuracyVerdict | None
    specialized_refusal_verdict: RefusalVerdict | None


def generate_accuracy_review(
    comparisons: Sequence[QuestionComparison], dataset: EvaluationDataset
) -> list[QuestionAccuracyReview]:
    """Build a review template from a real baseline-comparison run - verdicts left blank.

    Every field comes directly from a real `run_comparison()` result and
    the dataset's own hand-verified `note` - nothing here is invented.
    """
    note_by_question_id = {question.id: question.note for question in dataset.questions}
    reviews = []
    for comparison in comparisons:
        note = note_by_question_id.get(comparison.question_id)
        if note is None:
            raise AccuracyScoringError(
                f"No ground-truth note found for question_id="
                f"{comparison.question_id!r} in the evaluation dataset"
            )
        specialized_finding = comparison.specialized_finding
        reviews.append(
            QuestionAccuracyReview(
                question_id=comparison.question_id,
                question_text=comparison.question_text,
                ground_truth_note=note,
                baseline_claim=comparison.baseline_finding.claim,
                baseline_citations=tuple(
                    _citation_ref(citation)
                    for citation in comparison.baseline_finding.citations
                ),
                baseline_verdict=None,
                specialized_claim=(
                    specialized_finding.claim
                    if specialized_finding is not None
                    else None
                ),
                specialized_citations=(
                    tuple(
                        _citation_ref(citation)
                        for citation in specialized_finding.citations
                    )
                    if specialized_finding is not None
                    else ()
                ),
                specialized_error=comparison.specialized_error,
                specialized_verdict=None,
                specialized_refusal_verdict=None,
            )
        )
    return reviews


def _citation_ref_to_dict(citation: CitationRef) -> dict[str, object]:
    return {
        "document_extraction_id": citation.document_extraction_id,
        "page_number": citation.page_number,
        "supporting_text": citation.supporting_text,
    }


def _citation_ref_from_dict(payload: dict[str, object]) -> CitationRef:
    return CitationRef(
        document_extraction_id=payload["document_extraction_id"],  # type: ignore[arg-type]
        page_number=payload["page_number"],  # type: ignore[arg-type]
        supporting_text=payload["supporting_text"],  # type: ignore[arg-type]
    )


def _review_to_dict(review: QuestionAccuracyReview) -> dict[str, object]:
    return {
        "question_id": review.question_id,
        "question_text": review.question_text,
        "ground_truth_note": review.ground_truth_note,
        "baseline_claim": review.baseline_claim,
        "baseline_citations": [
            _citation_ref_to_dict(citation) for citation in review.baseline_citations
        ],
        "baseline_verdict": review.baseline_verdict,
        "specialized_claim": review.specialized_claim,
        "specialized_citations": [
            _citation_ref_to_dict(citation) for citation in review.specialized_citations
        ],
        "specialized_error": review.specialized_error,
        "specialized_verdict": review.specialized_verdict,
        "specialized_refusal_verdict": review.specialized_refusal_verdict,
    }


def save_accuracy_review(reviews: Sequence[QuestionAccuracyReview], path: Path) -> None:
    """Persist a review (template or completed) as indented JSON."""
    payload = {"reviews": [_review_to_dict(review) for review in reviews]}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_accuracy_review(path: Path) -> list[QuestionAccuracyReview]:
    """Parse a (possibly still-incomplete) accuracy review from JSON."""
    payload = json.loads(path.read_text())
    return [
        QuestionAccuracyReview(
            question_id=review["question_id"],
            question_text=review["question_text"],
            ground_truth_note=review["ground_truth_note"],
            baseline_claim=review["baseline_claim"],
            baseline_citations=tuple(
                _citation_ref_from_dict(citation)
                for citation in review["baseline_citations"]
            ),
            baseline_verdict=review["baseline_verdict"],
            specialized_claim=review["specialized_claim"],
            specialized_citations=tuple(
                _citation_ref_from_dict(citation)
                for citation in review["specialized_citations"]
            ),
            specialized_error=review["specialized_error"],
            specialized_verdict=review["specialized_verdict"],
            specialized_refusal_verdict=review["specialized_refusal_verdict"],
        )
        for review in payload["reviews"]
    ]


@dataclass(frozen=True)
class VerdictCounts:
    """Counts and proportions of each correctness verdict."""

    correct: int
    partially_correct: int
    incorrect: int

    @property
    def total(self) -> int:
        return self.correct + self.partially_correct + self.incorrect

    @property
    def correct_rate(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def partially_correct_rate(self) -> float:
        return self.partially_correct / self.total if self.total else 0.0

    @property
    def incorrect_rate(self) -> float:
        return self.incorrect / self.total if self.total else 0.0


@dataclass(frozen=True)
class RefusalCounts:
    """Counts and proportions of whether a refusal to answer was the right call."""

    appropriate: int
    inappropriate: int

    @property
    def total(self) -> int:
        return self.appropriate + self.inappropriate

    @property
    def appropriate_rate(self) -> float:
        return self.appropriate / self.total if self.total else 0.0


@dataclass(frozen=True)
class AccuracyReport:
    """Human-scored factual-accuracy comparison between both baselines.

    `specialized_claims` is scored only over questions where the
    specialized agent actually produced a claim; `specialized_refusals`
    covers the remainder separately, so a correct refusal is never
    conflated with an incorrect claim, and an incorrect refusal is never
    hidden inside a correctness rate that only ever measures claims made.
    """

    baseline: VerdictCounts
    specialized_claims: VerdictCounts
    specialized_refusals: RefusalCounts


def _validate_reviewed(reviews: Sequence[QuestionAccuracyReview]) -> None:
    """Fail closed on any question missing a verdict, rather than silently excluding it.

    The same discipline citation validation and human-review decisions
    already use elsewhere in this project: an incomplete review would
    otherwise quietly understate `total` and misrepresent the measured
    result as if every question had been considered.
    """
    problems = []
    for review in reviews:
        if review.baseline_verdict is None:
            problems.append(f"{review.question_id}: missing baseline_verdict")
        has_claim = review.specialized_claim is not None
        if has_claim and review.specialized_verdict is None:
            problems.append(f"{review.question_id}: missing specialized_verdict")
        if not has_claim and review.specialized_refusal_verdict is None:
            problems.append(
                f"{review.question_id}: missing specialized_refusal_verdict"
            )
        if has_claim and review.specialized_refusal_verdict is not None:
            problems.append(
                f"{review.question_id}: has both specialized_claim and "
                "specialized_refusal_verdict set"
            )
        if not has_claim and review.specialized_verdict is not None:
            problems.append(
                f"{review.question_id}: has specialized_verdict set but no "
                "specialized_claim"
            )
    if problems:
        raise AccuracyScoringError(
            "Accuracy review is incomplete or inconsistent:\n" + "\n".join(problems)
        )


def score_accuracy_review(reviews: Sequence[QuestionAccuracyReview]) -> AccuracyReport:
    """Aggregate a completed accuracy review into per-baseline verdict counts."""
    if not reviews:
        raise AccuracyScoringError("Accuracy review is empty")
    _validate_reviewed(reviews)

    baseline_verdicts = [review.baseline_verdict for review in reviews]
    specialized_verdicts = [
        review.specialized_verdict
        for review in reviews
        if review.specialized_verdict is not None
    ]
    refusal_verdicts = [
        review.specialized_refusal_verdict
        for review in reviews
        if review.specialized_refusal_verdict is not None
    ]

    return AccuracyReport(
        baseline=VerdictCounts(
            correct=sum(1 for v in baseline_verdicts if v == "correct"),
            partially_correct=sum(
                1 for v in baseline_verdicts if v == "partially_correct"
            ),
            incorrect=sum(1 for v in baseline_verdicts if v == "incorrect"),
        ),
        specialized_claims=VerdictCounts(
            correct=sum(1 for v in specialized_verdicts if v == "correct"),
            partially_correct=sum(
                1 for v in specialized_verdicts if v == "partially_correct"
            ),
            incorrect=sum(1 for v in specialized_verdicts if v == "incorrect"),
        ),
        specialized_refusals=RefusalCounts(
            appropriate=sum(1 for v in refusal_verdicts if v == "appropriate"),
            inappropriate=sum(1 for v in refusal_verdicts if v == "inappropriate"),
        ),
    )

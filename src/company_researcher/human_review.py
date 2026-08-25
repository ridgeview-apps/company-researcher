from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import HumanReview

ReviewDecision = Literal["approved", "edited", "rejected", "more_research_requested"]


class HumanReviewError(Exception):
    """Raised when a review decision cannot be recorded as requested."""


def needs_human_review(*, claim_type: str, evidence_sufficient: bool) -> bool:
    """Decide whether a finding must pause for human review before being treated as final.

    Deliberately narrow, per `docs/project-brief.md`'s fact/interpretation/
    significance framing: an interpretation (a judgement beyond what the
    evidence directly states, e.g. "this indicates governance instability")
    or an insufficiently evidenced claim both require review; a directly
    evidenced fact with sufficient evidence does not. No separate
    self-reported confidence score is used - this project has already found
    LLM self-assessment on a comparably subtle axis unreliable (see
    README.md's "A reverted attempt at citation entailment checking"), so
    the trigger is kept to two already-trusted, already-produced signals.
    """
    return claim_type == "interpretation" or not evidence_sufficient


def review_reason(*, claim_type: str, evidence_sufficient: bool) -> str:
    """Describe, deterministically, why a finding was flagged for review."""
    reasons = []
    if claim_type == "interpretation":
        reasons.append("claim_type=interpretation")
    if not evidence_sufficient:
        reasons.append("evidence_sufficient=false")
    return ", ".join(reasons)


async def record_pending_review(
    session: AsyncSession,
    *,
    company_number: str,
    question: str,
    generated_query: str,
    claim: str,
    claim_type: str,
    evidence_sufficient: bool,
    citations: list[dict[str, object]],
) -> int:
    """Persist a pending human review for a finding that needs one, returning its ID."""
    review = HumanReview(
        company_number=company_number,
        question=question,
        generated_query=generated_query,
        claim=claim,
        claim_type=claim_type,
        evidence_sufficient=evidence_sufficient,
        citations=citations,
        review_reason=review_reason(
            claim_type=claim_type, evidence_sufficient=evidence_sufficient
        ),
        status="pending",
    )
    session.add(review)
    await session.commit()
    return review.id


@dataclass(frozen=True)
class ReviewDecisionResult:
    """Summary of one applied human review decision, for CLI reporting."""

    review_id: int
    status: ReviewDecision
    claim: str


async def apply_review_decision(
    session: AsyncSession,
    review_id: int,
    decision: ReviewDecision,
    *,
    edited_claim: str | None = None,
    note: str | None = None,
    reviewer: str | None = None,
) -> ReviewDecisionResult:
    """Record a human analyst's decision against a pending review.

    Only ever applies to a review still in `pending` status - a review that
    has already been decided is not re-decided, the same fail-closed
    discipline this project already applies to citation validation, rather
    than silently overwriting a prior human decision.
    """
    review = await session.get(HumanReview, review_id)
    if review is None:
        raise HumanReviewError(f"No human review found with id={review_id}")
    if review.status != "pending":
        raise HumanReviewError(
            f"Human review {review_id} has already been decided "
            f"(status={review.status})"
        )
    if decision == "edited" and not edited_claim:
        raise HumanReviewError("An edited decision requires --edited-claim")

    review.status = decision
    review.decision_note = note
    review.reviewer = reviewer
    review.decided_at = datetime.now(UTC)
    if decision == "edited":
        review.edited_claim = edited_claim
    await session.commit()

    final_claim = edited_claim if decision == "edited" else review.claim
    assert final_claim is not None
    return ReviewDecisionResult(review_id=review_id, status=decision, claim=final_claim)

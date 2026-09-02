from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.db.models import HumanReview
from company_researcher.human_review import (
    HumanReviewError,
    ReviewDecision,
    apply_review_decision,
    get_review,
    list_reviews,
)

router = APIRouter(prefix="/reviews", tags=["reviews"])


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a request-scoped database session from the app's session factory."""
    session_factory = request.app.state.session_factory
    async with session_factory() as session:
        yield session


class CitationOut(BaseModel):
    document_extraction_id: int
    page_number: int
    supporting_text: str


class ReviewSummary(BaseModel):
    id: int
    status: str
    company_number: str
    question: str
    claim_type: str
    evidence_sufficient: bool
    review_reason: str
    created_at: datetime


class ReviewDetail(ReviewSummary):
    generated_query: str
    claim: str
    citations: list[CitationOut]
    edited_claim: str | None
    decision_note: str | None
    reviewer: str | None
    decided_at: datetime | None


class ReviewDecisionRequest(BaseModel):
    decision: ReviewDecision
    edited_claim: str | None = None
    note: str | None = None
    reviewer: str | None = None


class ReviewDecisionResponse(BaseModel):
    review_id: int
    status: ReviewDecision
    claim: str


def _to_summary(review: HumanReview) -> ReviewSummary:
    return ReviewSummary(
        id=review.id,
        status=review.status,
        company_number=review.company_number,
        question=review.question,
        claim_type=review.claim_type,
        evidence_sufficient=review.evidence_sufficient,
        review_reason=review.review_reason,
        created_at=review.created_at,
    )


def _to_detail(review: HumanReview) -> ReviewDetail:
    return ReviewDetail(
        **_to_summary(review).model_dump(),
        generated_query=review.generated_query,
        claim=review.claim,
        citations=[CitationOut(**citation) for citation in review.citations],
        edited_claim=review.edited_claim,
        decision_note=review.decision_note,
        reviewer=review.reviewer,
        decided_at=review.decided_at,
    )


@router.get("", response_model=list[ReviewSummary])
async def get_reviews(
    status: str | None = Query(default=None),
    session: AsyncSession = Depends(get_session),
) -> list[ReviewSummary]:
    """List persisted human reviews, optionally filtered by status."""
    reviews = await list_reviews(session, status=status)
    return [_to_summary(review) for review in reviews]


@router.get("/{review_id}", response_model=ReviewDetail)
async def get_review_detail(
    review_id: int,
    session: AsyncSession = Depends(get_session),
) -> ReviewDetail:
    """Fetch one human review, including its claim and cited evidence."""
    review = await get_review(session, review_id)
    if review is None:
        raise HTTPException(
            status_code=404, detail=f"No review found with id={review_id}"
        )
    return _to_detail(review)


@router.post("/{review_id}/decision", response_model=ReviewDecisionResponse)
async def decide_review(
    review_id: int,
    body: ReviewDecisionRequest,
    session: AsyncSession = Depends(get_session),
) -> ReviewDecisionResponse:
    """Record a human analyst's decision against one pending review."""
    review = await get_review(session, review_id)
    if review is None:
        raise HTTPException(
            status_code=404, detail=f"No review found with id={review_id}"
        )
    try:
        result = await apply_review_decision(
            session,
            review_id,
            body.decision,
            edited_claim=body.edited_claim,
            note=body.note,
            reviewer=body.reviewer,
        )
    except HumanReviewError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return ReviewDecisionResponse(
        review_id=result.review_id, status=result.status, claim=result.claim
    )

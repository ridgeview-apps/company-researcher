from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.config import Settings
from company_researcher.db.models import Company, HumanReview
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.human_review import (
    HumanReviewError,
    apply_review_decision,
    needs_human_review,
    record_pending_review,
    review_reason,
)

TEST_COMPANY_NUMBER = "TE000010"


def test_needs_human_review_is_true_for_an_interpretation() -> None:
    assert needs_human_review(claim_type="interpretation", evidence_sufficient=True)


def test_needs_human_review_is_true_for_insufficient_evidence() -> None:
    assert needs_human_review(claim_type="fact", evidence_sufficient=False)


def test_needs_human_review_is_false_for_a_sufficiently_evidenced_fact() -> None:
    assert not needs_human_review(claim_type="fact", evidence_sufficient=True)


def test_review_reason_combines_both_triggers() -> None:
    assert (
        review_reason(claim_type="interpretation", evidence_sufficient=False)
        == "claim_type=interpretation, evidence_sufficient=false"
    )


def test_review_reason_is_empty_when_no_trigger_applies() -> None:
    assert review_reason(claim_type="fact", evidence_sufficient=True) == ""


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
                delete(Company).where(Company.company_number == TEST_COMPANY_NUMBER)
            )
            await cleanup_session.commit()
        await engine.dispose()


@pytest_asyncio.fixture
async def company(session: AsyncSession) -> Company:
    company = Company(
        company_number=TEST_COMPANY_NUMBER,
        company_name="HUMAN REVIEW TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=datetime.now(UTC),
    )
    session.add(company)
    await session.commit()
    return company


@pytest.mark.asyncio
async def test_record_pending_review_persists_a_pending_row(
    session: AsyncSession, company: Company
) -> None:
    review_id = await record_pending_review(
        session,
        company_number=TEST_COMPANY_NUMBER,
        question="Does the evidence show governance instability?",
        generated_query="directors resignations",
        claim="This indicates governance instability.",
        claim_type="interpretation",
        evidence_sufficient=True,
        citations=[
            {"document_extraction_id": 1, "page_number": 2, "supporting_text": "quote"}
        ],
    )

    persisted = await session.get(HumanReview, review_id)
    assert persisted is not None
    assert persisted.status == "pending"
    assert persisted.review_reason == "claim_type=interpretation"
    assert persisted.citations == [
        {"document_extraction_id": 1, "page_number": 2, "supporting_text": "quote"}
    ]


@pytest.mark.asyncio
async def test_apply_review_decision_approves_a_pending_review(
    session: AsyncSession, company: Company
) -> None:
    review_id = await record_pending_review(
        session,
        company_number=TEST_COMPANY_NUMBER,
        question="Question",
        generated_query="query",
        claim="Original claim.",
        claim_type="fact",
        evidence_sufficient=False,
        citations=[],
    )

    result = await apply_review_decision(
        session, review_id, "approved", reviewer="alex"
    )

    assert result.status == "approved"
    assert result.claim == "Original claim."
    persisted = await session.get(HumanReview, review_id)
    assert persisted is not None
    assert persisted.status == "approved"
    assert persisted.reviewer == "alex"
    assert persisted.decided_at is not None


@pytest.mark.asyncio
async def test_apply_review_decision_edits_the_claim(
    session: AsyncSession, company: Company
) -> None:
    review_id = await record_pending_review(
        session,
        company_number=TEST_COMPANY_NUMBER,
        question="Question",
        generated_query="query",
        claim="Original claim.",
        claim_type="interpretation",
        evidence_sufficient=True,
        citations=[],
    )

    result = await apply_review_decision(
        session, review_id, "edited", edited_claim="Corrected claim."
    )

    assert result.claim == "Corrected claim."
    persisted = await session.get(HumanReview, review_id)
    assert persisted is not None
    assert persisted.edited_claim == "Corrected claim."
    assert persisted.claim == "Original claim."


@pytest.mark.asyncio
async def test_apply_review_decision_requires_an_edited_claim_for_edit(
    session: AsyncSession, company: Company
) -> None:
    review_id = await record_pending_review(
        session,
        company_number=TEST_COMPANY_NUMBER,
        question="Question",
        generated_query="query",
        claim="Original claim.",
        claim_type="interpretation",
        evidence_sufficient=True,
        citations=[],
    )

    with pytest.raises(HumanReviewError):
        await apply_review_decision(session, review_id, "edited")


@pytest.mark.asyncio
async def test_apply_review_decision_rejects_redeciding_an_already_decided_review(
    session: AsyncSession, company: Company
) -> None:
    review_id = await record_pending_review(
        session,
        company_number=TEST_COMPANY_NUMBER,
        question="Question",
        generated_query="query",
        claim="Original claim.",
        claim_type="fact",
        evidence_sufficient=False,
        citations=[],
    )
    await apply_review_decision(session, review_id, "approved")

    with pytest.raises(HumanReviewError):
        await apply_review_decision(session, review_id, "rejected")


@pytest.mark.asyncio
async def test_apply_review_decision_raises_for_an_unknown_review_id(
    session: AsyncSession,
) -> None:
    with pytest.raises(HumanReviewError):
        await apply_review_decision(session, 999_999_999, "approved")

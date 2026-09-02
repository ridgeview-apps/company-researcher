from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.config import Settings
from company_researcher.db.models import Company, HumanReview
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.main import create_app

TEST_COMPANY_NUMBER = "TE000011"


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
        company_name="API REVIEW TEST LIMITED",
        type="ltd",
        sic_codes=[],
        raw_profile={},
        retrieved_at=datetime.now(UTC),
    )
    session.add(company)
    await session.commit()
    return company


@pytest_asyncio.fixture
async def pending_review(session: AsyncSession, company: Company) -> HumanReview:
    review = HumanReview(
        company_number=TEST_COMPANY_NUMBER,
        question="Does the evidence show governance instability?",
        generated_query="directors resignations",
        claim="This indicates governance instability.",
        claim_type="interpretation",
        evidence_sufficient=True,
        citations=[
            {"document_extraction_id": 1, "page_number": 2, "supporting_text": "quote"}
        ],
        review_reason="claim_type=interpretation",
        status="pending",
    )
    session.add(review)
    await session.commit()
    return review


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(Settings(_env_file=None))  # type: ignore[call-arg]
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_get_reviews_lists_a_pending_review(
    client: TestClient, pending_review: HumanReview
) -> None:
    response = client.get("/reviews", params={"status": "pending"})

    assert response.status_code == 200
    ids = [review["id"] for review in response.json()]
    assert pending_review.id in ids


@pytest.mark.asyncio
async def test_get_reviews_filters_by_status(
    client: TestClient, pending_review: HumanReview
) -> None:
    response = client.get("/reviews", params={"status": "approved"})

    assert response.status_code == 200
    ids = [review["id"] for review in response.json()]
    assert pending_review.id not in ids


@pytest.mark.asyncio
async def test_get_review_detail_includes_citations(
    client: TestClient, pending_review: HumanReview
) -> None:
    response = client.get(f"/reviews/{pending_review.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["claim"] == "This indicates governance instability."
    assert body["citations"] == [
        {"document_extraction_id": 1, "page_number": 2, "supporting_text": "quote"}
    ]


@pytest.mark.asyncio
async def test_get_review_detail_404s_for_an_unknown_id(client: TestClient) -> None:
    response = client.get("/reviews/999999999")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_decide_review_approves_a_pending_review(
    client: TestClient, pending_review: HumanReview
) -> None:
    response = client.post(
        f"/reviews/{pending_review.id}/decision",
        json={"decision": "approved", "reviewer": "test-analyst"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "review_id": pending_review.id,
        "status": "approved",
        "claim": "This indicates governance instability.",
    }

    detail = client.get(f"/reviews/{pending_review.id}").json()
    assert detail["status"] == "approved"
    assert detail["reviewer"] == "test-analyst"


@pytest.mark.asyncio
async def test_decide_review_rejects_a_second_decision(
    client: TestClient, pending_review: HumanReview
) -> None:
    first = client.post(
        f"/reviews/{pending_review.id}/decision", json={"decision": "approved"}
    )
    assert first.status_code == 200

    second = client.post(
        f"/reviews/{pending_review.id}/decision", json={"decision": "rejected"}
    )
    assert second.status_code == 400


@pytest.mark.asyncio
async def test_decide_review_404s_for_an_unknown_id(client: TestClient) -> None:
    response = client.post("/reviews/999999999/decision", json={"decision": "approved"})

    assert response.status_code == 404

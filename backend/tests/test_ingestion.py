from collections.abc import AsyncIterator, Callable, Coroutine

import httpx2
import pytest
import pytest_asyncio
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from company_researcher.companies_house.client import CompaniesHouseClient
from company_researcher.config import Settings
from company_researcher.db.models import Company, Filing
from company_researcher.db.session import create_database_engine, create_session_factory
from company_researcher.ingestion import ingest_company

TEST_COMPANY_NUMBER = "TE000001"
AsyncMockHandler = Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]


def _create_client(handler: AsyncMockHandler) -> CompaniesHouseClient:
    return CompaniesHouseClient(
        api_key="test-api-key",
        base_url="https://example.test",
        transport=httpx2.MockTransport(handler),
    )


def _profile_payload(company_status: str) -> dict[str, object]:
    return {
        "company_name": "TEST INGESTION LIMITED",
        "company_number": TEST_COMPANY_NUMBER,
        "type": "ltd",
        "company_status": company_status,
        "sic_codes": ["62012"],
    }


def _filing_history_payload(
    transaction_ids: list[str],
    *,
    include_document_links: bool = True,
) -> dict[str, object]:
    return {
        "items": [
            {
                "transaction_id": transaction_id,
                "category": "accounts",
                "date": "2025-01-01",
                "description": "accounts-with-accounts-type-full",
                "type": "AA",
                "links": (
                    {
                        "document_metadata": (
                            "https://document-api.company-information.service.gov.uk/"
                            f"document/{transaction_id}-document"
                        )
                    }
                    if include_document_links
                    else {}
                ),
            }
            for transaction_id in transaction_ids
        ],
        "items_per_page": 100,
        "start_index": 0,
        "total_count": len(transaction_ids),
    }


def _handler_for(
    company_status: str,
    transaction_ids: list[str],
    *,
    include_document_links: bool = True,
) -> AsyncMockHandler:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/filing-history"):
            return httpx2.Response(
                200,
                json=_filing_history_payload(
                    transaction_ids,
                    include_document_links=include_document_links,
                ),
            )
        return httpx2.Response(200, json=_profile_payload(company_status))

    return handler


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


@pytest.mark.asyncio
async def test_ingest_company_persists_profile_and_filings(
    session: AsyncSession,
) -> None:
    async with _create_client(
        _handler_for("active", ["transaction-1", "transaction-2"])
    ) as client:
        result = await ingest_company(session, client, TEST_COMPANY_NUMBER)

    assert result.company_number == TEST_COMPANY_NUMBER
    assert result.filing_count == 2

    company = await session.get(Company, TEST_COMPANY_NUMBER)
    assert company is not None
    assert company.company_status == "active"

    filings = (
        await session.execute(
            select(Filing).where(Filing.company_number == TEST_COMPANY_NUMBER)
        )
    ).all()
    assert len(filings) == 2
    assert filings[0][0].source_document_id is not None
    assert filings[0][0].document_metadata_url is not None


@pytest.mark.asyncio
async def test_ingest_company_is_idempotent(session: AsyncSession) -> None:
    async with _create_client(
        _handler_for("active", ["transaction-1", "transaction-2"])
    ) as client:
        await ingest_company(session, client, TEST_COMPANY_NUMBER)

    async with _create_client(
        _handler_for("dissolved", ["transaction-1", "transaction-2"])
    ) as client:
        result = await ingest_company(session, client, TEST_COMPANY_NUMBER)

    assert result.filing_count == 2

    filings = (
        await session.execute(
            select(Filing).where(Filing.company_number == TEST_COMPANY_NUMBER)
        )
    ).all()
    assert len(filings) == 2

    company = await session.get(Company, TEST_COMPANY_NUMBER)
    assert company is not None
    assert company.company_status == "dissolved"


@pytest.mark.asyncio
async def test_ingest_company_allows_filing_without_document(
    session: AsyncSession,
) -> None:
    async with _create_client(
        _handler_for(
            "active",
            ["transaction-without-document"],
            include_document_links=False,
        )
    ) as client:
        await ingest_company(session, client, TEST_COMPANY_NUMBER)

    filing = (
        await session.execute(
            select(Filing).where(
                Filing.company_number == TEST_COMPANY_NUMBER,
                Filing.transaction_id == "transaction-without-document",
            )
        )
    ).scalar_one()
    assert filing.source_document_id is None
    assert filing.document_metadata_url is None

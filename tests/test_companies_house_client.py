import base64
from collections.abc import Callable, Coroutine

import httpx2
import pytest

from company_researcher.companies_house.client import CompaniesHouseClient
from company_researcher.companies_house.exceptions import (
    CompaniesHouseAuthenticationError,
    CompaniesHouseConfigurationError,
    CompaniesHouseNotFoundError,
    CompaniesHouseRateLimitError,
    CompaniesHouseResponseError,
)
from company_researcher.config import Settings

SyncMockHandler = Callable[[httpx2.Request], httpx2.Response]
AsyncMockHandler = Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
MockHandler = SyncMockHandler | AsyncMockHandler


def create_client(handler: MockHandler) -> CompaniesHouseClient:
    return CompaniesHouseClient(
        api_key="test-api-key",
        base_url="https://example.test",
        transport=httpx2.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_get_company_profile_authenticates_and_normalizes_number() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        expected_credentials = base64.b64encode(b"test-api-key:").decode()
        assert request.headers["authorization"] == f"Basic {expected_credentials}"
        assert request.url.path == "/company/00000006"
        return httpx2.Response(
            200,
            json={
                "company_name": "EXAMPLE LIMITED",
                "company_number": "00000006",
                "type": "ltd",
                "company_status": "active",
                "future_api_field": "preserved",
            },
        )

    async with create_client(handler) as client:
        profile = await client.get_company_profile("6")

    assert profile.company_number == "00000006"
    assert profile.company_name == "EXAMPLE LIMITED"
    assert profile.model_extra == {"future_api_field": "preserved"}


@pytest.mark.asyncio
async def test_get_filing_history_fetches_every_page() -> None:
    requested_start_indexes: list[int] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        start_index = int(request.url.params["start_index"])
        requested_start_indexes.append(start_index)
        item_numbers = [1, 2] if start_index == 0 else [3]
        return httpx2.Response(
            200,
            json={
                "items": [
                    {
                        "transaction_id": f"transaction-{item_number}",
                        "category": "accounts",
                        "date": "2025-01-01",
                        "description": "accounts-with-accounts-type-full",
                        "type": "AA",
                    }
                    for item_number in item_numbers
                ],
                "items_per_page": 2,
                "start_index": start_index,
                "total_count": 3,
            },
        )

    async with create_client(handler) as client:
        history = await client.get_filing_history("00000006", page_size=2)

    assert requested_start_indexes == [0, 2]
    assert history.total_count == 3
    assert [item.transaction_id for item in history.items] == [
        "transaction-1",
        "transaction-2",
        "transaction-3",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, CompaniesHouseAuthenticationError),
        (404, CompaniesHouseNotFoundError),
        (429, CompaniesHouseRateLimitError),
    ],
)
async def test_get_company_profile_maps_http_errors(
    status_code: int,
    expected_error: type[Exception],
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status_code, request=request)

    async with create_client(handler) as client:
        with pytest.raises(expected_error):
            await client.get_company_profile("00000006")


@pytest.mark.asyncio
async def test_get_company_profile_rejects_malformed_payload() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={"company_number": "00000006"},
            request=request,
        )

    async with create_client(handler) as client:
        with pytest.raises(CompaniesHouseResponseError):
            await client.get_company_profile("00000006")


def test_from_settings_requires_api_key() -> None:
    settings = Settings(companies_house_api_key=None)

    with pytest.raises(CompaniesHouseConfigurationError):
        CompaniesHouseClient.from_settings(settings)

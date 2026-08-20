import base64

import httpx2
import pytest

from company_researcher.companies_house.document_client import (
    CompaniesHouseDocumentClient,
)
from company_researcher.companies_house.exceptions import (
    CompaniesHouseConfigurationError,
    CompaniesHouseResponseError,
)
from company_researcher.config import Settings


def create_client(handler: httpx2.MockTransport) -> CompaniesHouseDocumentClient:
    return CompaniesHouseDocumentClient(
        api_key="test-api-key",
        base_url="https://document.example.test",
        transport=handler,
    )


@pytest.mark.asyncio
async def test_get_document_metadata_authenticates_and_validates_response() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        expected_credentials = base64.b64encode(b"test-api-key:").decode()
        assert request.headers["authorization"] == f"Basic {expected_credentials}"
        assert request.headers["accept"] == "application/json"
        assert request.url.path == "/document/document-123"
        return httpx2.Response(
            200,
            json={
                "id": "document-123",
                "created_at": "2025-01-02T03:04:05Z",
                "etag": "document-etag",
                "pages": 12,
                "resources": {
                    "application/pdf": {
                        "content_length": 12345,
                        "created_at": "2025-01-02T03:04:05Z",
                    }
                },
                "future_api_field": "preserved",
            },
        )

    async with create_client(httpx2.MockTransport(handler)) as client:
        metadata = await client.get_document_metadata(" document-123 ")

    assert metadata.id == "document-123"
    assert metadata.pages == 12
    assert metadata.resources["application/pdf"].content_length == 12345
    assert metadata.model_extra == {"future_api_field": "preserved"}


@pytest.mark.asyncio
async def test_get_document_metadata_rejects_malformed_payload() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"id": "document-123"}, request=request)

    async with create_client(httpx2.MockTransport(handler)) as client:
        with pytest.raises(CompaniesHouseResponseError):
            await client.get_document_metadata("document-123")


@pytest.mark.parametrize("document_id", ["", "   ", "path/segment"])
@pytest.mark.asyncio
async def test_get_document_metadata_rejects_invalid_document_id(
    document_id: str,
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        pytest.fail("Invalid document IDs must fail before making a request")

    async with create_client(httpx2.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="document_id"):
            await client.get_document_metadata(document_id)


def test_from_settings_requires_api_key() -> None:
    settings = Settings(companies_house_api_key=None)

    with pytest.raises(CompaniesHouseConfigurationError):
        CompaniesHouseDocumentClient.from_settings(settings)

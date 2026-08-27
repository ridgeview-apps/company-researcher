from dataclasses import dataclass
from typing import Self

from company_researcher.companies_house._base_client import CompaniesHouseBaseClient
from company_researcher.companies_house.exceptions import (
    CompaniesHouseConfigurationError,
    CompaniesHouseResponseError,
)
from company_researcher.companies_house.models import DocumentMetadata
from company_researcher.config import Settings


@dataclass(frozen=True)
class DocumentContent:
    """A downloaded filing document and its verified representation."""

    document_id: str
    media_type: str
    content: bytes


class CompaniesHouseDocumentClient(CompaniesHouseBaseClient):
    """Async client for the Companies House Document API."""

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Create a document client from validated application settings."""
        if settings.companies_house_api_key is None:
            raise CompaniesHouseConfigurationError(
                "COMPANIES_HOUSE_API_KEY is required to call Companies House"
            )

        return cls(
            api_key=settings.companies_house_api_key.get_secret_value(),
            base_url=str(settings.companies_house_document_base_url),
        )

    async def get_document_metadata(self, document_id: str) -> DocumentMetadata:
        """Fetch metadata describing a filing document's representations."""
        normalized_id = self._normalize_document_id(document_id)

        return await self._get_model(
            f"document/{normalized_id}",
            DocumentMetadata,
        )

    async def get_document_content(self, document_id: str) -> DocumentContent:
        """Download a filing document in PDF format."""
        normalized_id = self._normalize_document_id(document_id)
        response = await self._get_response(
            f"document/{normalized_id}/content",
            headers={"Accept": "application/pdf"},
            follow_redirects=True,
        )
        media_type = response.headers.get("content-type", "").partition(";")[0]
        media_type = media_type.strip().lower()
        if media_type != "application/pdf":
            raise CompaniesHouseResponseError(
                "Companies House returned an unexpected document content type"
            )
        if not response.content:
            raise CompaniesHouseResponseError(
                "Companies House returned an empty document"
            )

        return DocumentContent(
            document_id=normalized_id,
            media_type=media_type,
            content=response.content,
        )

    @staticmethod
    def _normalize_document_id(document_id: str) -> str:
        normalized_id = document_id.strip()
        if not normalized_id or "/" in normalized_id:
            raise ValueError("document_id must be a non-empty path segment")
        return normalized_id

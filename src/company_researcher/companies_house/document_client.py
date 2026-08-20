from typing import Self

from company_researcher.companies_house._base_client import CompaniesHouseBaseClient
from company_researcher.companies_house.exceptions import (
    CompaniesHouseConfigurationError,
)
from company_researcher.companies_house.models import DocumentMetadata
from company_researcher.config import Settings


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
        normalized_id = document_id.strip()
        if not normalized_id or "/" in normalized_id:
            raise ValueError("document_id must be a non-empty path segment")

        return await self._get_model(
            f"document/{normalized_id}",
            DocumentMetadata,
        )

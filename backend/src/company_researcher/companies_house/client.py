import re
from typing import Self

from company_researcher.companies_house._base_client import CompaniesHouseBaseClient
from company_researcher.companies_house.exceptions import (
    CompaniesHouseConfigurationError,
    CompaniesHouseResponseError,
)
from company_researcher.companies_house.models import (
    CompanyProfile,
    FilingHistory,
    FilingHistoryPage,
)
from company_researcher.config import Settings

COMPANY_NUMBER_PATTERN = re.compile(r"^[A-Z0-9]{8}$")


def normalize_company_number(company_number: str) -> str:
    """Normalize and validate a Companies House company number."""
    normalized_number = company_number.strip().upper()
    if normalized_number.isdecimal():
        normalized_number = normalized_number.zfill(8)
    if not COMPANY_NUMBER_PATTERN.fullmatch(normalized_number):
        raise ValueError("company_number must contain 8 letters or digits")
    return normalized_number


class CompaniesHouseClient(CompaniesHouseBaseClient):
    """Async client for the Companies House Public Data REST API."""

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Create a client from validated application settings."""
        if settings.companies_house_api_key is None:
            raise CompaniesHouseConfigurationError(
                "COMPANIES_HOUSE_API_KEY is required to call Companies House"
            )

        return cls(
            api_key=settings.companies_house_api_key.get_secret_value(),
            base_url=str(settings.companies_house_base_url),
        )

    async def get_company_profile(self, company_number: str) -> CompanyProfile:
        """Fetch a company's current structured profile."""
        normalized_number = normalize_company_number(company_number)
        return await self._get_model(
            f"company/{normalized_number}",
            CompanyProfile,
        )

    async def get_filing_history_page(
        self,
        company_number: str,
        *,
        start_index: int = 0,
        items_per_page: int = 100,
    ) -> FilingHistoryPage:
        """Fetch one filing-history page."""
        if start_index < 0:
            raise ValueError("start_index must not be negative")
        if not 1 <= items_per_page <= 100:
            raise ValueError("items_per_page must be between 1 and 100")

        normalized_number = normalize_company_number(company_number)
        return await self._get_model(
            f"company/{normalized_number}/filing-history",
            FilingHistoryPage,
            params={
                "items_per_page": items_per_page,
                "start_index": start_index,
            },
        )

    async def get_filing_history(
        self,
        company_number: str,
        *,
        page_size: int = 100,
    ) -> FilingHistory:
        """Fetch every filing-history page for a company."""
        items = []
        start_index = 0
        total_count: int | None = None

        while total_count is None or start_index < total_count:
            page = await self.get_filing_history_page(
                company_number,
                start_index=start_index,
                items_per_page=page_size,
            )
            if total_count is None:
                total_count = page.total_count

            items.extend(page.items)
            next_start_index = page.start_index + len(page.items)

            if next_start_index <= start_index and next_start_index < total_count:
                raise CompaniesHouseResponseError(
                    "Filing-history pagination did not advance"
                )
            start_index = next_start_index

        return FilingHistory(items=items, total_count=total_count)

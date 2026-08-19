import re
from types import TracebackType
from typing import Self, TypeVar

import httpx2
from pydantic import BaseModel, ValidationError

from company_researcher.companies_house.exceptions import (
    CompaniesHouseAuthenticationError,
    CompaniesHouseConfigurationError,
    CompaniesHouseConnectionError,
    CompaniesHouseNotFoundError,
    CompaniesHouseRateLimitError,
    CompaniesHouseResponseError,
)
from company_researcher.companies_house.models import (
    CompanyProfile,
    FilingHistory,
    FilingHistoryPage,
)
from company_researcher.config import Settings

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
COMPANY_NUMBER_PATTERN = re.compile(r"^[A-Z0-9]{8}$")


class CompaniesHouseClient:
    """Async client for the Companies House Public Data REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: float = 10.0,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise CompaniesHouseConfigurationError(
                "COMPANIES_HOUSE_API_KEY must not be empty"
            )

        self._client = httpx2.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            auth=httpx2.BasicAuth(api_key, ""),
            headers={
                "Accept": "application/json",
                "User-Agent": "company-researcher/0.1",
            },
            timeout=timeout_seconds,
            transport=transport,
        )

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

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close network connections owned by the client."""
        await self._client.aclose()

    async def get_company_profile(self, company_number: str) -> CompanyProfile:
        """Fetch a company's current structured profile."""
        normalized_number = self._normalize_company_number(company_number)
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

        normalized_number = self._normalize_company_number(company_number)
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

    async def _get_model(
        self,
        path: str,
        model_type: type[ResponseModel],
        *,
        params: dict[str, int] | None = None,
    ) -> ResponseModel:
        try:
            response = await self._client.get(path, params=params)
        except httpx2.RequestError as error:
            raise CompaniesHouseConnectionError(
                "Could not connect to Companies House"
            ) from error

        self._raise_for_status(response)

        try:
            payload: object = response.json()
            return model_type.model_validate(payload)
        except (ValueError, ValidationError) as error:
            raise CompaniesHouseResponseError(
                "Companies House returned an invalid response payload"
            ) from error

    @staticmethod
    def _raise_for_status(response: httpx2.Response) -> None:
        if response.status_code < 400:
            return
        if response.status_code == 401:
            raise CompaniesHouseAuthenticationError(
                "Companies House rejected the API key"
            )
        if response.status_code == 404:
            raise CompaniesHouseNotFoundError("Companies House resource was not found")
        if response.status_code == 429:
            raise CompaniesHouseRateLimitError("Companies House rate limit exceeded")
        raise CompaniesHouseResponseError(
            f"Companies House returned HTTP {response.status_code}"
        )

    @staticmethod
    def _normalize_company_number(company_number: str) -> str:
        normalized_number = company_number.strip().upper()
        if normalized_number.isdecimal():
            normalized_number = normalized_number.zfill(8)
        if not COMPANY_NUMBER_PATTERN.fullmatch(normalized_number):
            raise ValueError("company_number must contain 8 letters or digits")
        return normalized_number

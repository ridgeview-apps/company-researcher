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

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class CompaniesHouseBaseClient:
    """Shared authenticated transport for Companies House API services."""

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

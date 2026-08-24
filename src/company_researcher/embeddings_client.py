from collections.abc import Sequence
from types import TracebackType
from typing import Self

import httpx2
from pydantic import BaseModel, ValidationError

from company_researcher.config import Settings


class EmbeddingsError(Exception):
    """Base exception for embeddings-provider integration failures."""


class EmbeddingsConfigurationError(EmbeddingsError):
    """Raised when embeddings-provider configuration is incomplete."""


class EmbeddingsConnectionError(EmbeddingsError):
    """Raised when the embeddings provider cannot be reached."""


class EmbeddingsAuthenticationError(EmbeddingsError):
    """Raised when the embeddings provider rejects the API key."""


class EmbeddingsRateLimitError(EmbeddingsError):
    """Raised when the embeddings provider's request quota is exhausted."""


class EmbeddingsResponseError(EmbeddingsError):
    """Raised for an unexpected status or invalid response payload."""


class _EmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class _EmbeddingsResponsePayload(BaseModel):
    data: list[_EmbeddingItem]


class EmbeddingsClient:
    """Async client for an OpenAI-compatible embeddings REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key.strip():
            raise EmbeddingsConfigurationError("OPENAI_API_KEY must not be empty")

        self._model = model
        self._client = httpx2.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "company-researcher/0.1",
            },
            timeout=timeout_seconds,
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Create a client from validated application settings."""
        if settings.openai_api_key is None:
            raise EmbeddingsConfigurationError(
                "OPENAI_API_KEY is required to call the embeddings provider"
            )

        return cls(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=str(settings.openai_base_url),
            model=settings.openai_embedding_model,
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

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input in the same order."""
        if not texts:
            raise ValueError("texts must not be empty")

        try:
            response = await self._client.post(
                "embeddings", json={"model": self._model, "input": list(texts)}
            )
        except httpx2.RequestError as error:
            raise EmbeddingsConnectionError(
                "Could not connect to the embeddings provider"
            ) from error

        self._raise_for_status(response)

        try:
            payload = _EmbeddingsResponsePayload.model_validate(response.json())
        except (ValueError, ValidationError) as error:
            raise EmbeddingsResponseError(
                "Embeddings provider returned an invalid response payload"
            ) from error

        if len(payload.data) != len(texts):
            raise EmbeddingsResponseError(
                "Embeddings provider returned "
                f"{len(payload.data)} embedding(s) for {len(texts)} text(s)"
            )

        ordered_items = sorted(payload.data, key=lambda item: item.index)
        return [item.embedding for item in ordered_items]

    @staticmethod
    def _raise_for_status(response: httpx2.Response) -> None:
        if response.status_code < 400:
            return
        if response.status_code == 401:
            raise EmbeddingsAuthenticationError(
                "Embeddings provider rejected the API key"
            )
        if response.status_code == 429:
            raise EmbeddingsRateLimitError("Embeddings provider rate limit exceeded")
        raise EmbeddingsResponseError(
            f"Embeddings provider returned HTTP {response.status_code}"
        )

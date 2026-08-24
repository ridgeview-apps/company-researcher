from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol, Self

import httpx2
from pydantic import BaseModel, ValidationError

from company_researcher.config import Settings

ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    """One message in a chat completion request."""

    role: ChatRole
    content: str


class ChatProvider(Protocol):
    """Boundary for single-turn chat completion."""

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Complete a chat, returning the assistant's response text."""
        ...


class ChatError(Exception):
    """Base exception for chat-provider integration failures."""


class ChatConfigurationError(ChatError):
    """Raised when chat-provider configuration is incomplete."""


class ChatConnectionError(ChatError):
    """Raised when the chat provider cannot be reached."""


class ChatAuthenticationError(ChatError):
    """Raised when the chat provider rejects the API key."""


class ChatRateLimitError(ChatError):
    """Raised when the chat provider's request quota is exhausted."""


class ChatResponseError(ChatError):
    """Raised for an unexpected status or invalid response payload."""


class _ChatChoiceMessage(BaseModel):
    content: str | None = None


class _ChatChoice(BaseModel):
    message: _ChatChoiceMessage


class _ChatCompletionResponsePayload(BaseModel):
    choices: list[_ChatChoice]


class ChatClient:
    """Async client for an OpenAI-compatible chat completions REST API."""

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
            raise ChatConfigurationError("OPENAI_API_KEY must not be empty")

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
            raise ChatConfigurationError(
                "OPENAI_API_KEY is required to call the chat provider"
            )

        return cls(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=str(settings.openai_base_url),
            model=settings.openai_chat_model,
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

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Complete a chat, returning the assistant's response text."""
        if not messages:
            raise ValueError("messages must not be empty")

        try:
            response = await self._client.post(
                "chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in messages
                    ],
                },
            )
        except httpx2.RequestError as error:
            raise ChatConnectionError(
                "Could not connect to the chat provider"
            ) from error

        self._raise_for_status(response)

        try:
            payload = _ChatCompletionResponsePayload.model_validate(response.json())
        except (ValueError, ValidationError) as error:
            raise ChatResponseError(
                "Chat provider returned an invalid response payload"
            ) from error

        if not payload.choices or payload.choices[0].message.content is None:
            raise ChatResponseError("Chat provider returned no completion content")

        return payload.choices[0].message.content

    @staticmethod
    def _raise_for_status(response: httpx2.Response) -> None:
        if response.status_code < 400:
            return
        if response.status_code == 401:
            raise ChatAuthenticationError("Chat provider rejected the API key")
        if response.status_code == 429:
            raise ChatRateLimitError("Chat provider rate limit exceeded")
        raise ChatResponseError(f"Chat provider returned HTTP {response.status_code}")

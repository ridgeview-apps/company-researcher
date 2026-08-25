from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol, Self, TypeVar

import httpx2
from pydantic import BaseModel, ValidationError

from company_researcher.config import Settings

ChatRole = Literal["system", "user", "assistant"]

StructuredResponse = TypeVar("StructuredResponse", bound=BaseModel)


@dataclass(frozen=True)
class ChatMessage:
    """One message in a chat completion request."""

    role: ChatRole
    content: str


@dataclass(frozen=True)
class ChatUsage:
    """Token usage the provider reported for one completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatProvider(Protocol):
    """Boundary for single-turn chat completion."""

    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        """Complete a chat, returning the assistant's response text."""
        ...

    async def complete_structured(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[StructuredResponse],
    ) -> StructuredResponse:
        """Complete a chat, validating the response against `response_model`."""
        ...


class UsageAwareChatProvider(Protocol):
    """Boundary for chat completion that also reports token usage.

    A separate protocol from `ChatProvider` rather than an addition to it,
    so callers that only need `complete`/`complete_structured` (and their
    test fakes, in particular `investigation_agent.py`'s original
    single-question path before usage tracking existed) are unaffected by
    this capability. Bundles both methods the same way `ChatProvider`
    itself does, even though a given call site may only ever use one of
    them - `investigation_agent.py` needs both (query generation calls
    `complete_with_usage`, synthesis calls `complete_structured_with_usage`),
    and `baseline_agent.py` only needs the structured one, but a single
    shared protocol is simpler than two near-duplicates.
    """

    async def complete_with_usage(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[str, ChatUsage | None]:
        """Complete a chat, returning the response text alongside token usage."""
        ...

    async def complete_structured_with_usage(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[StructuredResponse],
    ) -> tuple[StructuredResponse, ChatUsage | None]:
        """Complete a structured chat, returning the result alongside token usage."""
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


class _ChatUsagePayload(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _ChatCompletionResponsePayload(BaseModel):
    choices: list[_ChatChoice]
    usage: _ChatUsagePayload | None = None


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
        content, _usage = await self.complete_with_usage(messages)
        return content

    async def complete_with_usage(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[str, ChatUsage | None]:
        """Complete a chat, returning the response text alongside token usage.

        A separate method from `complete` rather than a change to it, so the
        `ChatProvider` protocol and every existing caller (in particular
        `investigation_agent.py`, which only needs the text) stay untouched.
        Usage is `None` if the provider's response omits it.
        """
        return await self._request_completion(messages)

    async def complete_structured(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[StructuredResponse],
    ) -> StructuredResponse:
        """Complete a chat, validating the response against `response_model`.

        Uses the provider's strict JSON-schema structured-output mode, built
        from `response_model.model_json_schema()`. Strict mode requires every
        object in the schema to set `additionalProperties: false`, which
        Pydantic only emits when a model's `model_config` sets
        `extra="forbid"` — that is the caller's responsibility, not this
        client's.
        """
        result, _usage = await self.complete_structured_with_usage(
            messages, response_model
        )
        return result

    async def complete_structured_with_usage(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[StructuredResponse],
    ) -> tuple[StructuredResponse, ChatUsage | None]:
        """Complete a structured chat, returning the result alongside token usage.

        See `complete_with_usage` for why this is a separate method rather
        than a change to `complete_structured`.
        """
        content, usage = await self._request_completion(
            messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
        )
        try:
            return response_model.model_validate_json(content), usage
        except ValidationError as error:
            raise ChatResponseError(
                "Chat provider returned content that does not match the expected schema"
            ) from error

    async def _request_completion(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, object] | None = None,
    ) -> tuple[str, ChatUsage | None]:
        """Send one chat completion request and return the response text and usage."""
        if not messages:
            raise ValueError("messages must not be empty")

        body: dict[str, object] = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        }
        if response_format is not None:
            body["response_format"] = response_format

        try:
            response = await self._client.post("chat/completions", json=body)
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

        usage = (
            ChatUsage(
                prompt_tokens=payload.usage.prompt_tokens,
                completion_tokens=payload.usage.completion_tokens,
                total_tokens=payload.usage.total_tokens,
            )
            if payload.usage is not None
            else None
        )
        return payload.choices[0].message.content, usage

    @staticmethod
    def _raise_for_status(response: httpx2.Response) -> None:
        if response.status_code < 400:
            return
        if response.status_code == 401:
            raise ChatAuthenticationError("Chat provider rejected the API key")
        if response.status_code == 429:
            raise ChatRateLimitError("Chat provider rate limit exceeded")
        raise ChatResponseError(f"Chat provider returned HTTP {response.status_code}")

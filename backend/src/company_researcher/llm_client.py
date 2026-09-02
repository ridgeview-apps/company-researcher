import json
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol, Self, TypeVar

import httpx2
from langsmith import traceable
from pydantic import BaseModel, ValidationError

from company_researcher.config import Settings

ChatRole = Literal["system", "user", "assistant", "tool"]

StructuredResponse = TypeVar("StructuredResponse", bound=BaseModel)


@dataclass(frozen=True)
class ToolCall:
    """One function-call invocation the model requested."""

    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ChatMessage:
    """One message in a chat completion request.

    `tool_calls` and `tool_call_id` are only ever set for the two message
    shapes a tool-calling round trip needs beyond a plain
    system/user/assistant turn: an assistant message that requested one or
    more tool calls (`tool_calls` set, `content` often empty), and a `tool`
    role message reporting one call's result back (`tool_call_id` set,
    matching the `ToolCall.id` it answers). Every other message leaves both
    `None`, so existing callers that only ever pass `role`/`content` are
    unaffected.
    """

    role: ChatRole
    content: str
    tool_calls: tuple[ToolCall, ...] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    """One function-calling tool a chat completion may be offered."""

    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True)
class ToolCallTurn:
    """One assistant turn in a tool-calling loop.

    `tool_calls` is empty exactly when the model chose to respond with
    final text instead of calling another tool - the loop-ending signal a
    caller like `tool_baseline_agent.py` checks for.
    """

    content: str | None
    tool_calls: tuple[ToolCall, ...]


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


class ToolAwareChatProvider(Protocol):
    """Boundary for a tool-calling loop: one tool-offering turn plus a final structured turn.

    A separate protocol from `UsageAwareChatProvider`, not an addition to
    it, so callers that never offer tools (every existing caller) are
    unaffected. `tool_baseline_agent.py` is the only current caller: it
    drives `complete_with_tools_and_usage` in a loop until the model stops
    requesting tools, then calls `complete_structured_with_usage` once more
    for the final `Finding`.
    """

    async def complete_with_tools_and_usage(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
    ) -> tuple[ToolCallTurn, ChatUsage | None]:
        """Complete one turn of a tool-calling loop, returning it alongside token usage."""
        ...

    async def complete_structured_with_usage(
        self,
        messages: Sequence[ChatMessage],
        response_model: type[StructuredResponse],
    ) -> tuple[StructuredResponse, ChatUsage | None]:
        """Complete a structured chat, returning the result alongside token usage."""
        ...


class FullChatProvider(UsageAwareChatProvider, ToolAwareChatProvider, Protocol):
    """Boundary bundling every capability `baseline_comparison.py` needs.

    `compare_question` runs all three answer paths (no-tool baseline,
    tool-using baseline, specialized agent) against one shared client, so
    it needs both `UsageAwareChatProvider` and `ToolAwareChatProvider`'s
    methods available on a single parameter type rather than accepting two
    separately-typed client arguments for what is, in practice, one
    `ChatClient` instance.
    """


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


class _ChatToolCallFunctionPayload(BaseModel):
    name: str
    arguments: str


class _ChatToolCallPayload(BaseModel):
    id: str
    function: _ChatToolCallFunctionPayload


class _ChatChoiceMessage(BaseModel):
    content: str | None = None
    tool_calls: list[_ChatToolCallPayload] | None = None


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

    async def complete_with_tools_and_usage(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
    ) -> tuple[ToolCallTurn, ChatUsage | None]:
        """Complete one turn of a tool-calling loop, returning it alongside token usage.

        A separate request path from `_request_completion` rather than an
        extension of it: that method treats a `None` response `content` as
        an error, which is the normal, expected shape of a turn where the
        model requests tool calls instead of answering directly.
        """
        return await self._request_completion_with_tools(messages, tools)

    @traceable(run_type="llm", name="chat_completion")
    async def _request_completion(
        self,
        messages: Sequence[ChatMessage],
        *,
        response_format: dict[str, object] | None = None,
    ) -> tuple[str, ChatUsage | None]:
        """Send one chat completion request and return the response text and usage.

        `@traceable` is a no-op unless `LANGSMITH_TRACING`/`LANGSMITH_API_KEY`
        are set in the process environment (see `cli.py`'s
        `_configure_langsmith_tracing`, which bridges `Settings` into those
        variables); when active, it reports this call - the messages sent,
        the raw response, and latency - as its own span nested under
        whichever LangGraph node invoked it, since `self` is automatically
        excluded from the captured inputs and this is the sole real network
        call every `complete*` method funnels through.
        """
        if not messages:
            raise ValueError("messages must not be empty")

        body: dict[str, object] = {
            "model": self._model,
            "messages": [self._serialize_message(message) for message in messages],
        }
        if response_format is not None:
            body["response_format"] = response_format

        payload = await self._post_chat_completion(body)

        if not payload.choices or payload.choices[0].message.content is None:
            raise ChatResponseError("Chat provider returned no completion content")

        return payload.choices[0].message.content, self._parse_usage(payload)

    @traceable(run_type="llm", name="chat_completion_with_tools")
    async def _request_completion_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition],
    ) -> tuple[ToolCallTurn, ChatUsage | None]:
        """Send one tool-offering chat completion request and return the resulting turn."""
        if not messages:
            raise ValueError("messages must not be empty")
        if not tools:
            raise ValueError("tools must not be empty")

        body: dict[str, object] = {
            "model": self._model,
            "messages": [self._serialize_message(message) for message in messages],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ],
        }

        payload = await self._post_chat_completion(body)

        if not payload.choices:
            raise ChatResponseError("Chat provider returned no completion content")

        message = payload.choices[0].message
        tool_calls = tuple(
            ToolCall(
                id=raw_call.id,
                name=raw_call.function.name,
                arguments=json.loads(raw_call.function.arguments),
            )
            for raw_call in (message.tool_calls or [])
        )
        return ToolCallTurn(
            content=message.content, tool_calls=tool_calls
        ), self._parse_usage(payload)

    async def _post_chat_completion(
        self, body: dict[str, object]
    ) -> _ChatCompletionResponsePayload:
        """Send one raw chat completion request and validate its response shape."""
        try:
            response = await self._client.post("chat/completions", json=body)
        except httpx2.RequestError as error:
            raise ChatConnectionError(
                "Could not connect to the chat provider"
            ) from error

        self._raise_for_status(response)

        try:
            return _ChatCompletionResponsePayload.model_validate(response.json())
        except (ValueError, ValidationError) as error:
            raise ChatResponseError(
                "Chat provider returned an invalid response payload"
            ) from error

    @staticmethod
    def _serialize_message(message: ChatMessage) -> dict[str, object]:
        """Serialize one `ChatMessage` into the OpenAI-compatible request shape."""
        serialized: dict[str, object] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_calls:
            serialized["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments),
                    },
                }
                for tool_call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            serialized["tool_call_id"] = message.tool_call_id
        return serialized

    @staticmethod
    def _parse_usage(payload: _ChatCompletionResponsePayload) -> ChatUsage | None:
        if payload.usage is None:
            return None
        return ChatUsage(
            prompt_tokens=payload.usage.prompt_tokens,
            completion_tokens=payload.usage.completion_tokens,
            total_tokens=payload.usage.total_tokens,
        )

    @staticmethod
    def _raise_for_status(response: httpx2.Response) -> None:
        if response.status_code < 400:
            return
        if response.status_code == 401:
            raise ChatAuthenticationError("Chat provider rejected the API key")
        if response.status_code == 429:
            raise ChatRateLimitError("Chat provider rate limit exceeded")
        raise ChatResponseError(f"Chat provider returned HTTP {response.status_code}")

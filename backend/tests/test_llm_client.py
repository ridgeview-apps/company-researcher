import json
from collections.abc import Callable, Coroutine

import httpx2
import pytest
from pydantic import BaseModel, ConfigDict

from company_researcher.config import Settings
from company_researcher.llm_client import (
    ChatAuthenticationError,
    ChatClient,
    ChatConfigurationError,
    ChatMessage,
    ChatRateLimitError,
    ChatResponseError,
    ChatUsage,
    ToolCall,
    ToolDefinition,
)


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    reason: str


SyncMockHandler = Callable[[httpx2.Request], httpx2.Response]
AsyncMockHandler = Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
MockHandler = SyncMockHandler | AsyncMockHandler


def create_client(handler: MockHandler) -> ChatClient:
    return ChatClient(
        api_key="test-api-key",
        base_url="https://example.test",
        model="gpt-4o-mini",
        transport=httpx2.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_complete_authenticates_and_returns_message_content() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["authorization"] == "Bearer test-api-key"
        assert request.url.path == "/chat/completions"
        assert json.loads(request.content) == {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hello"},
            ],
        }
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hi there."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async with create_client(handler) as client:
        result = await client.complete(
            [
                ChatMessage(role="system", content="Be terse."),
                ChatMessage(role="user", content="Hello"),
            ]
        )

    assert result == "Hi there."


@pytest.mark.asyncio
async def test_complete_rejects_empty_messages() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("should not send a request for empty messages")

    async with create_client(handler) as client:
        with pytest.raises(ValueError, match="messages must not be empty"):
            await client.complete([])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, ChatAuthenticationError),
        (429, ChatRateLimitError),
        (500, ChatResponseError),
    ],
)
async def test_complete_maps_http_errors(
    status_code: int,
    expected_error: type[Exception],
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status_code, request=request)

    async with create_client(handler) as client:
        with pytest.raises(expected_error):
            await client.complete([ChatMessage(role="user", content="Hello")])


@pytest.mark.asyncio
async def test_complete_rejects_malformed_payload() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"unexpected": "shape"}, request=request)

    async with create_client(handler) as client:
        with pytest.raises(ChatResponseError):
            await client.complete([ChatMessage(role="user", content="Hello")])


@pytest.mark.asyncio
async def test_complete_rejects_empty_choices() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"choices": []}, request=request)

    async with create_client(handler) as client:
        with pytest.raises(ChatResponseError):
            await client.complete([ChatMessage(role="user", content="Hello")])


@pytest.mark.asyncio
async def test_complete_rejects_null_content() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": None}}]},
            request=request,
        )

    async with create_client(handler) as client:
        with pytest.raises(ChatResponseError):
            await client.complete([ChatMessage(role="user", content="Hello")])


@pytest.mark.asyncio
async def test_complete_structured_sends_strict_json_schema_and_validates_result() -> (
    None
):
    async def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "_Verdict",
                "strict": True,
                "schema": _Verdict.model_json_schema(),
            },
        }
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"approved": True, "reason": "Evidence is sufficient."}
                            ),
                        }
                    }
                ]
            },
        )

    async with create_client(handler) as client:
        verdict = await client.complete_structured(
            [ChatMessage(role="user", content="Is this approved?")], _Verdict
        )

    assert verdict == _Verdict(approved=True, reason="Evidence is sufficient.")


@pytest.mark.asyncio
async def test_complete_structured_rejects_content_that_fails_validation() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": json.dumps({})}}
                ]
            },
            request=request,
        )

    async with create_client(handler) as client:
        with pytest.raises(ChatResponseError):
            await client.complete_structured(
                [ChatMessage(role="user", content="Is this approved?")], _Verdict
            )


@pytest.mark.asyncio
async def test_complete_with_usage_returns_token_counts_when_present() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Hi there."}}],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            },
            request=request,
        )

    async with create_client(handler) as client:
        content, usage = await client.complete_with_usage(
            [ChatMessage(role="user", content="Hello")]
        )

    assert content == "Hi there."
    assert usage == ChatUsage(prompt_tokens=12, completion_tokens=4, total_tokens=16)


@pytest.mark.asyncio
async def test_complete_with_usage_returns_none_when_provider_omits_it() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "Hi."}}]},
            request=request,
        )

    async with create_client(handler) as client:
        _content, usage = await client.complete_with_usage(
            [ChatMessage(role="user", content="Hello")]
        )

    assert usage is None


@pytest.mark.asyncio
async def test_complete_structured_with_usage_returns_result_and_token_counts() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"approved": True, "reason": "Evidence is sufficient."}
                            ),
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 8,
                    "total_tokens": 38,
                },
            },
            request=request,
        )

    async with create_client(handler) as client:
        verdict, usage = await client.complete_structured_with_usage(
            [ChatMessage(role="user", content="Is this approved?")], _Verdict
        )

    assert verdict == _Verdict(approved=True, reason="Evidence is sufficient.")
    assert usage == ChatUsage(prompt_tokens=30, completion_tokens=8, total_tokens=38)


def test_from_settings_requires_api_key() -> None:
    settings = Settings(openai_api_key=None)

    with pytest.raises(ChatConfigurationError):
        ChatClient.from_settings(settings)


_GET_WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    },
)


@pytest.mark.asyncio
async def test_complete_with_tools_sends_tool_definitions_and_returns_requested_calls() -> (
    None
):
    async def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        assert body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": _GET_WEATHER_TOOL.parameters,
                },
            }
        ]
        assert "response_format" not in body
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": json.dumps({"city": "London"}),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 6,
                    "total_tokens": 26,
                },
            },
            request=request,
        )

    async with create_client(handler) as client:
        turn, usage = await client.complete_with_tools_and_usage(
            [ChatMessage(role="user", content="What's the weather in London?")],
            [_GET_WEATHER_TOOL],
        )

    assert turn.content is None
    assert turn.tool_calls == (
        ToolCall(id="call-1", name="get_weather", arguments={"city": "London"}),
    )
    assert usage == ChatUsage(prompt_tokens=20, completion_tokens=6, total_tokens=26)


@pytest.mark.asyncio
async def test_complete_with_tools_returns_no_tool_calls_when_model_answers_directly() -> (
    None
):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "It's sunny."}}
                ]
            },
            request=request,
        )

    async with create_client(handler) as client:
        turn, _usage = await client.complete_with_tools_and_usage(
            [ChatMessage(role="user", content="What's the weather in London?")],
            [_GET_WEATHER_TOOL],
        )

    assert turn.content == "It's sunny."
    assert turn.tool_calls == ()


@pytest.mark.asyncio
async def test_complete_with_tools_rejects_empty_messages_and_empty_tools() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("should not send a request")

    async with create_client(handler) as client:
        with pytest.raises(ValueError, match="messages must not be empty"):
            await client.complete_with_tools_and_usage([], [_GET_WEATHER_TOOL])
        with pytest.raises(ValueError, match="tools must not be empty"):
            await client.complete_with_tools_and_usage(
                [ChatMessage(role="user", content="Hello")], []
            )


@pytest.mark.asyncio
async def test_complete_with_tools_serializes_a_tool_round_trip_correctly() -> None:
    """A prior assistant tool-call message and its tool-result reply serialize to the expected shape."""

    async def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        assert body["messages"] == [
            {"role": "user", "content": "What's the weather in London?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "London"}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": json.dumps({"temperature_c": 18}),
                "tool_call_id": "call-1",
            },
        ]
        return httpx2.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "It's 18C."}}]
            },
            request=request,
        )

    async with create_client(handler) as client:
        turn, _usage = await client.complete_with_tools_and_usage(
            [
                ChatMessage(role="user", content="What's the weather in London?"),
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=(
                        ToolCall(
                            id="call-1",
                            name="get_weather",
                            arguments={"city": "London"},
                        ),
                    ),
                ),
                ChatMessage(
                    role="tool",
                    content=json.dumps({"temperature_c": 18}),
                    tool_call_id="call-1",
                ),
            ],
            [_GET_WEATHER_TOOL],
        )

    assert turn.content == "It's 18C."

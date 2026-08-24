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


def test_from_settings_requires_api_key() -> None:
    settings = Settings(openai_api_key=None)

    with pytest.raises(ChatConfigurationError):
        ChatClient.from_settings(settings)

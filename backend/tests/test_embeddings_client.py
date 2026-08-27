import json
from collections.abc import Callable, Coroutine

import httpx2
import pytest

from company_researcher.config import Settings
from company_researcher.embeddings_client import (
    EmbeddingsAuthenticationError,
    EmbeddingsClient,
    EmbeddingsConfigurationError,
    EmbeddingsRateLimitError,
    EmbeddingsResponseError,
)

SyncMockHandler = Callable[[httpx2.Request], httpx2.Response]
AsyncMockHandler = Callable[[httpx2.Request], Coroutine[None, None, httpx2.Response]]
MockHandler = SyncMockHandler | AsyncMockHandler


def create_client(handler: MockHandler) -> EmbeddingsClient:
    return EmbeddingsClient(
        api_key="test-api-key",
        base_url="https://example.test",
        model="text-embedding-3-small",
        transport=httpx2.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_embed_authenticates_and_returns_vectors_in_request_order() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.headers["authorization"] == "Bearer test-api-key"
        assert request.url.path == "/embeddings"
        assert json.loads(request.content) == {
            "model": "text-embedding-3-small",
            "input": ["alpha", "bravo"],
        }
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.2, 0.3]},
                    {"object": "embedding", "index": 0, "embedding": [0.0, 0.1]},
                ],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    async with create_client(handler) as client:
        vectors = await client.embed(["alpha", "bravo"])

    assert vectors == [[0.0, 0.1], [0.2, 0.3]]


@pytest.mark.asyncio
async def test_embed_rejects_empty_input() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("should not send a request for empty input")

    async with create_client(handler) as client:
        with pytest.raises(ValueError, match="texts must not be empty"):
            await client.embed([])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, EmbeddingsAuthenticationError),
        (429, EmbeddingsRateLimitError),
        (500, EmbeddingsResponseError),
    ],
)
async def test_embed_maps_http_errors(
    status_code: int,
    expected_error: type[Exception],
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status_code, request=request)

    async with create_client(handler) as client:
        with pytest.raises(expected_error):
            await client.embed(["alpha"])


@pytest.mark.asyncio
async def test_embed_rejects_malformed_payload() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"unexpected": "shape"}, request=request)

    async with create_client(handler) as client:
        with pytest.raises(EmbeddingsResponseError):
            await client.embed(["alpha"])


@pytest.mark.asyncio
async def test_embed_rejects_mismatched_embedding_count() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.0]}]},
            request=request,
        )

    async with create_client(handler) as client:
        with pytest.raises(EmbeddingsResponseError):
            await client.embed(["alpha", "bravo"])


def test_from_settings_requires_api_key() -> None:
    settings = Settings(openai_api_key=None)

    with pytest.raises(EmbeddingsConfigurationError):
        EmbeddingsClient.from_settings(settings)

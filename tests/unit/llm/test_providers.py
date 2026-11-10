"""Adapter normalization, safe errors, retries, and redaction tests."""

from __future__ import annotations

import logging

import pytest

from app.llm.errors import LLMProviderError
from app.llm.google import GoogleProvider
from app.llm.openai_compatible import LlamaCompatibleProvider, NvidiaProvider
from app.llm.rate_limit import AsyncRateLimiter
from app.schemas.providers import LLMErrorCode, ProviderKind
from tests.unit.llm.conftest import (
    StubClient,
    json_response,
    make_request,
    timeout_error,
)

API_KEY = "synthetic-api-key-not-real"


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def nvidia(client: StubClient, *, retries: int = 2) -> NvidiaProvider:
    return NvidiaProvider(
        model="synthetic-nvidia-model",
        endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
        api_key=API_KEY,
        timeout_seconds=5,
        max_retries=retries,
        retry_base_delay_seconds=0,
        rate_limiter=AsyncRateLimiter(60_000),
        client=client,  # type: ignore[arg-type]
        sleep=SleepRecorder(),
    )


def google(client: StubClient, *, retries: int = 2) -> GoogleProvider:
    return GoogleProvider(
        model="synthetic-google-model",
        endpoint=(
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/synthetic-google-model:generateContent"
        ),
        api_key=API_KEY,
        timeout_seconds=5,
        max_retries=retries,
        retry_base_delay_seconds=0,
        rate_limiter=AsyncRateLimiter(60_000),
        client=client,  # type: ignore[arg-type]
        sleep=SleepRecorder(),
    )


def llama(client: StubClient) -> LlamaCompatibleProvider:
    return LlamaCompatibleProvider(
        model="synthetic-llama-model",
        endpoint="https://llama.example.invalid/v1/chat/completions",
        api_key=API_KEY,
        timeout_seconds=5,
        max_retries=0,
        retry_base_delay_seconds=0,
        rate_limiter=AsyncRateLimiter(60_000),
        client=client,  # type: ignore[arg-type]
        auth_header="X-Host-Key",
        auth_scheme="Token",
    )


async def test_nvidia_sends_openai_shape_and_normalizes_response() -> None:
    client = StubClient(
        [
            json_response(
                200,
                {
                    "id": "chatcmpl-synthetic",
                    "model": "synthetic-nvidia-model-v2",
                    "choices": [
                        {
                            "message": {"content": "  Potassium is above range.  "},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )
        ]
    )

    response = await nvidia(client).generate(make_request())

    assert client.calls[0]["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert client.calls[0]["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert client.calls[0]["json"]["model"] == "synthetic-nvidia-model"
    assert client.calls[0]["json"]["messages"][0]["role"] == "system"
    assert client.calls[0]["json"]["stream"] is False
    assert response.provider is ProviderKind.NVIDIA
    assert response.model == "synthetic-nvidia-model"
    assert response.reported_model == "synthetic-nvidia-model-v2"
    assert response.text == "Potassium is above range."
    assert response.usage.total_tokens == 15
    assert response.finish_reason == "stop"


async def test_google_sends_generate_content_shape_and_normalizes_response() -> None:
    client = StubClient(
        [
            json_response(
                200,
                {
                    "responseId": "google-response-1",
                    "modelVersion": "synthetic-google-model-002",
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "First. "}, {"text": "Second."}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 12,
                        "candidatesTokenCount": 7,
                        "totalTokenCount": 19,
                    },
                },
            )
        ]
    )

    response = await google(client).generate(make_request())

    call = client.calls[0]
    assert call["headers"]["x-goog-api-key"] == API_KEY
    assert "Authorization" not in call["headers"]
    assert call["json"]["systemInstruction"]["parts"][0]["text"]
    assert call["json"]["contents"][0]["role"] == "user"
    assert call["json"]["generationConfig"]["maxOutputTokens"] == 256
    assert response.provider is ProviderKind.GOOGLE
    assert response.text == "First. Second."
    assert response.usage.total_tokens == 19
    assert response.response_id == "google-response-1"


async def test_llama_uses_custom_auth_header_and_openai_format() -> None:
    client = StubClient(
        [
            json_response(
                200,
                {
                    "choices": [{"message": {"content": "Synthetic."}, "finish_reason": "stop"}]
                },
            )
        ]
    )

    response = await llama(client).generate(make_request())

    assert client.calls[0]["headers"]["X-Host-Key"] == f"Token {API_KEY}"
    assert "Authorization" not in client.calls[0]["headers"]
    assert response.provider is ProviderKind.LLAMA


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": "not-a-list"},
        {"choices": []},
        {"choices": [{"message": {"content": 42}}]},
    ],
)
async def test_malformed_success_bodies_raise_safe_errors(body: dict[str, object]) -> None:
    client = StubClient([json_response(200, body)])

    with pytest.raises(LLMProviderError) as raised:
        await nvidia(client).generate(make_request())

    assert raised.value.code is LLMErrorCode.MALFORMED_RESPONSE
    assert "synthetic" not in str(raised.value)


async def test_empty_completion_is_rejected() -> None:
    client = StubClient(
        [
            json_response(
                200,
                {"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]},
            )
        ]
    )

    with pytest.raises(LLMProviderError) as raised:
        await nvidia(client).generate(make_request())

    assert raised.value.code is LLMErrorCode.EMPTY_RESPONSE


async def test_google_safety_block_is_a_safe_error() -> None:
    client = StubClient(
        [json_response(200, {"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []})]
    )

    with pytest.raises(LLMProviderError) as raised:
        await google(client).generate(make_request())

    assert raised.value.code is LLMErrorCode.BLOCKED_RESPONSE
    assert raised.value.retryable is False


async def test_authentication_failure_is_not_retried_or_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = StubClient(
        [
            json_response(401, {"error": {"message": f"bad key {API_KEY}"}}),
            json_response(200, {"choices": [{"message": {"content": "unused"}}]}),
        ]
    )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(LLMProviderError) as raised:
            await nvidia(client).generate(make_request())

    assert raised.value.code is LLMErrorCode.AUTHENTICATION_FAILED
    assert raised.value.retryable is False
    assert len(client.calls) == 1
    assert API_KEY not in caplog.text
    assert "Interpret the potassium result" not in caplog.text


async def test_retryable_statuses_are_bounded() -> None:
    client = StubClient(
        [
            json_response(503, {"error": "unavailable"}),
            json_response(503, {"error": "unavailable"}),
            json_response(503, {"error": "unavailable"}),
            json_response(200, {"choices": [{"message": {"content": "unused"}}]}),
        ]
    )
    provider = nvidia(client, retries=2)

    with pytest.raises(LLMProviderError) as raised:
        await provider.generate(make_request())

    assert raised.value.code is LLMErrorCode.UNAVAILABLE
    assert len(client.calls) == 3


async def test_timeout_is_retried_then_succeeds() -> None:
    client = StubClient(
        [
            timeout_error(),
            json_response(
                200,
                {"choices": [{"message": {"content": "Recovered."}, "finish_reason": "stop"}]},
            ),
        ]
    )
    sleep = SleepRecorder()
    provider = NvidiaProvider(
        model="synthetic-model",
        endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
        api_key=API_KEY,
        timeout_seconds=5,
        max_retries=1,
        retry_base_delay_seconds=0.25,
        rate_limiter=AsyncRateLimiter(60_000),
        client=client,  # type: ignore[arg-type]
        sleep=sleep,
    )

    response = await provider.generate(make_request())

    assert response.text == "Recovered."
    assert len(client.calls) == 2
    assert sleep.delays == [0.25]


async def test_exhausted_timeouts_raise_timeout() -> None:
    client = StubClient([timeout_error(), timeout_error()])
    provider = nvidia(client, retries=1)

    with pytest.raises(LLMProviderError) as raised:
        await provider.generate(make_request())

    assert raised.value.code is LLMErrorCode.TIMEOUT
    assert len(client.calls) == 2

"""Synthetic HTTP and provider fixtures for external adapter tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from app.llm.errors import LLMProviderError
from app.schemas.privacy import PreparedPayload, QuestionType
from app.schemas.providers import LLMMessage, LLMRequest, LLMResponse, ProviderKind


class StubClient:
    """Deterministic httpx.AsyncClient stand-in that records request arguments."""

    def __init__(self, outcomes: Sequence[httpx.Response | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        # The stub mirrors httpx.AsyncClient.post, whose signature has this name.
        timeout: httpx.Timeout,  # noqa: ASYNC109
    ) -> httpx.Response:
        self.calls.append(
            {"url": url, "headers": dict(headers), "json": json, "timeout": timeout}
        )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        self.closed = True


def json_response(status_code: int, payload: dict[str, Any]) -> httpx.Response:
    """Build a response with the request attached, as httpx normally does."""

    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return httpx.Response(status_code, json=payload, request=request)


def timeout_error() -> httpx.ReadTimeout:
    """Build a real httpx timeout for retry-path tests."""

    return httpx.ReadTimeout(
        "synthetic timeout", request=httpx.Request("POST", "https://example.invalid")
    )


def make_payload() -> PreparedPayload:
    """Build a minimal approved payload without invoking the privacy gate."""

    return PreparedPayload(
        request_id="req-llm-001",
        question="Interpret the potassium result.",
        question_type=QuestionType.LAB_VALUE,
    )


def make_request(
    *,
    payload: PreparedPayload | None = None,
    request_id: str = "req-llm-001",
) -> LLMRequest:
    """Build a normalized request with an explicit instruction/data separation."""

    resolved = payload or make_payload()
    return LLMRequest(
        request_id=request_id,
        prompt_id="reasoning",
        prompt_version="v1",
        payload=resolved,
        messages=(
            LLMMessage(role="system", content="You explain evidence only."),
            LLMMessage(role="user", content="Explain the approved payload."),
        ),
        max_output_tokens=256,
        temperature=0.2,
    )


class FakeProvider:
    """Provider protocol stand-in that never performs network I/O."""

    def __init__(
        self,
        *,
        kind: ProviderKind,
        model: str,
        endpoint: str,
        response: LLMResponse | None = None,
        error: LLMProviderError | None = None,
    ) -> None:
        self.kind = kind
        self.model = model
        self.endpoint = endpoint
        self._response = response
        self._error = error
        self.calls = 0
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response

    async def aclose(self) -> None:
        return None


def make_response(
    *,
    kind: ProviderKind = ProviderKind.NVIDIA,
    model: str = "synthetic-model",
    request_id: str = "req-llm-001",
    text: str = "Synthetic explanation.",
) -> LLMResponse:
    """Build a normalized completion for router tests."""

    return LLMResponse(
        request_id=request_id,
        provider=kind,
        model=model,
        text=text,
        latency_ms=5,
    )


@pytest.fixture
def payload() -> PreparedPayload:
    return make_payload()

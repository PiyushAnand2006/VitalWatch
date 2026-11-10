"""Shared HTTP behavior for isolated provider adapters."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import httpx

from app.llm.errors import LLMProviderError
from app.llm.rate_limit import AsyncRateLimiter
from app.schemas.providers import LLMErrorCode, LLMRequest, LLMResponse, ProviderKind

logger = logging.getLogger(__name__)

MAX_RESPONSE_BYTES = 2_000_000
MAX_BACKOFF_SECONDS = 10.0
RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


class LLMProvider(Protocol):
    """Minimal provider contract consumed by the router."""

    kind: ProviderKind
    model: str
    endpoint: str

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Convert and send one approved request, returning a normalized response."""

        ...

    async def aclose(self) -> None:
        """Release any owned HTTP resources."""

        ...


class BaseHTTPProvider(ABC):
    """Bounded, log-safe HTTP transport with per-provider rate limiting."""

    kind: ProviderKind

    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        api_key: str | None,
        timeout_seconds: float,
        max_retries: int,
        retry_base_delay_seconds: float,
        rate_limiter: AsyncRateLimiter,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay_seconds
        self._rate_limiter = rate_limiter
        self._client = client
        self._owns_client = client is None
        self._sleep = sleep

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Send a bounded request sequence; failures are normalized and unsent text is safe."""

        await self._rate_limiter.acquire()
        started = time.perf_counter()
        last_error: LLMProviderError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._post(request)
            except LLMProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self._max_retries:
                    raise
                await self._sleep(self._backoff_seconds(attempt, exc.retry_after_seconds))
                continue
            latency_ms = max(0, int((time.perf_counter() - started) * 1000))
            parsed = self._parse(response, request, latency_ms)
            logger.info(
                "external model call completed provider=%s model=%s latency_ms=%d "
                "total_tokens=%d",
                self.kind.value,
                self.model,
                latency_ms,
                parsed.usage.total_tokens,
            )
            return parsed
        raise last_error or LLMProviderError(LLMErrorCode.UNAVAILABLE)

    async def aclose(self) -> None:
        """Close only a client created by this adapter."""

        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post(self, request: LLMRequest) -> httpx.Response:
        client = self._get_client()
        try:
            response = await client.post(
                self.endpoint,
                headers=self._headers(),
                json=self._body(request),
                timeout=httpx.Timeout(self._timeout_seconds),
            )
        except httpx.TimeoutException as exc:
            raise LLMProviderError(LLMErrorCode.TIMEOUT, retryable=True) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(LLMErrorCode.NETWORK_ERROR, retryable=True) from exc
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise LLMProviderError(LLMErrorCode.MALFORMED_RESPONSE)
        if response.status_code >= 300:
            self._raise_status_error(response)
        return response

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(follow_redirects=False)
        return self._client

    def _raise_status_error(self, response: httpx.Response) -> None:
        status = response.status_code
        logger.warning(
            "external model call failed provider=%s model=%s status=%d",
            self.kind.value,
            self.model,
            status,
        )
        if status in {401, 403}:
            raise LLMProviderError(
                LLMErrorCode.AUTHENTICATION_FAILED, status_code=status
            )
        if status == 429:
            raise LLMProviderError(
                LLMErrorCode.RATE_LIMITED,
                retryable=True,
                status_code=status,
                retry_after_seconds=_retry_after(response),
            )
        if status in RETRYABLE_STATUSES:
            raise LLMProviderError(
                LLMErrorCode.UNAVAILABLE, retryable=True, status_code=status
            )
        raise LLMProviderError(LLMErrorCode.BAD_REQUEST, status_code=status)

    def _parse(
        self, response: httpx.Response, request: LLMRequest, latency_ms: int
    ) -> LLMResponse:
        try:
            data: Any = response.json()
        except ValueError as exc:
            raise LLMProviderError(LLMErrorCode.MALFORMED_RESPONSE) from exc
        if not isinstance(data, dict):
            raise LLMProviderError(LLMErrorCode.MALFORMED_RESPONSE)
        try:
            return self._normalize(data, request, latency_ms)
        except LLMProviderError:
            raise
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMProviderError(LLMErrorCode.MALFORMED_RESPONSE) from exc

    def _backoff_seconds(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return float(min(MAX_BACKOFF_SECONDS, max(0.0, retry_after)))
        return float(
            min(MAX_BACKOFF_SECONDS, self._retry_base_delay * float(2**attempt))
        )

    @abstractmethod
    def _headers(self) -> dict[str, str]:
        """Build authentication headers without logging their values."""

    @abstractmethod
    def _body(self, request: LLMRequest) -> dict[str, Any]:
        """Convert the normalized request to the host-specific JSON body."""

    @abstractmethod
    def _normalize(
        self, data: dict[str, Any], request: LLMRequest, latency_ms: int
    ) -> LLMResponse:
        """Normalize a host response or raise a safe malformed-response error."""


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(str(value))
    except ValueError:
        return None

"""OpenAI-compatible chat adapter used by NVIDIA and configurable Llama hosts."""

from __future__ import annotations

from typing import Any

from app.llm.base import BaseHTTPProvider
from app.llm.errors import LLMProviderError
from app.schemas.providers import (
    LLMErrorCode,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderKind,
)


class OpenAICompatibleProvider(BaseHTTPProvider):
    """Normalize the documented OpenAI chat-completions request/response shape."""

    kind: ProviderKind

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _body(self, request: LLMRequest) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }

    def _normalize(
        self, data: dict[str, Any], request: LLMRequest, latency_ms: int
    ) -> LLMResponse:
        choices = data["choices"]
        if not isinstance(choices, list) or not choices:
            raise LLMProviderError(LLMErrorCode.MALFORMED_RESPONSE)
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMProviderError(LLMErrorCode.MALFORMED_RESPONSE)
        message = choice["message"]
        if not isinstance(message, dict):
            raise LLMProviderError(LLMErrorCode.MALFORMED_RESPONSE)
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise LLMProviderError(LLMErrorCode.MALFORMED_RESPONSE)
        text = content if isinstance(content, str) else ""
        if not text.strip():
            raise LLMProviderError(LLMErrorCode.EMPTY_RESPONSE)
        usage = data.get("usage")
        normalized_usage = LLMUsage()
        if isinstance(usage, dict):
            normalized_usage = LLMUsage(
                prompt_tokens=_nonnegative_int(usage.get("prompt_tokens")),
                completion_tokens=_nonnegative_int(usage.get("completion_tokens")),
                total_tokens=_nonnegative_int(usage.get("total_tokens")),
            )
        finish_reason = choice.get("finish_reason")
        response_id = data.get("id")
        reported_model = data.get("model")
        return LLMResponse(
            request_id=request.request_id,
            provider=self.kind,
            model=self.model,
            text=text.strip(),
            finish_reason=finish_reason if isinstance(finish_reason, str) else None,
            usage=normalized_usage,
            response_id=response_id if isinstance(response_id, str) else None,
            reported_model=reported_model if isinstance(reported_model, str) else None,
            latency_ms=latency_ms,
        )


class NvidiaProvider(OpenAICompatibleProvider):
    """NVIDIA hosted NIM chat completions; endpoint and model are configured."""

    kind = ProviderKind.NVIDIA


class LlamaCompatibleProvider(OpenAICompatibleProvider):
    """Configurable Llama-compatible host using a documented OpenAI chat API."""

    kind = ProviderKind.LLAMA

    def __init__(
        self,
        *,
        auth_header: str,
        auth_scheme: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._auth_header = auth_header
        self._auth_scheme = auth_scheme

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key and self._auth_header:
            value = (
                f"{self._auth_scheme} {self._api_key}"
                if self._auth_scheme
                else self._api_key
            )
            headers[self._auth_header] = value
        return headers


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)

"""Google Gemini generateContent adapter using the documented REST contract."""

from __future__ import annotations

from typing import Any

from app.llm.base import BaseHTTPProvider, nonnegative_int
from app.llm.errors import LLMProviderError
from app.schemas.providers import (
    LLMErrorCode,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderKind,
)


class GoogleProvider(BaseHTTPProvider):
    """Normalize Gemini's generateContent request and response shapes."""

    kind = ProviderKind.GOOGLE

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["x-goog-api-key"] = self._api_key
        return headers

    def _body(self, request: LLMRequest) -> dict[str, Any]:
        system_message, *user_messages = request.messages
        return {
            "systemInstruction": {"parts": [{"text": system_message.content}]},
            "contents": [
                {"role": "user", "parts": [{"text": message.content}]}
                for message in user_messages
            ],
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_output_tokens,
            },
        }

    def _normalize(
        self, data: dict[str, Any], request: LLMRequest, latency_ms: int
    ) -> LLMResponse:
        prompt_feedback = data.get("promptFeedback")
        if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
            raise LLMProviderError(LLMErrorCode.BLOCKED_RESPONSE)
        candidates = data["candidates"]
        if not isinstance(candidates, list) or not candidates:
            raise LLMProviderError(LLMErrorCode.MALFORMED_RESPONSE)
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise LLMProviderError(LLMErrorCode.MALFORMED_RESPONSE)
        if candidate.get("finishReason") in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"}:
            raise LLMProviderError(LLMErrorCode.BLOCKED_RESPONSE)
        content = candidate["content"]
        if not isinstance(content, dict):
            raise LLMProviderError(LLMErrorCode.MALFORMED_RESPONSE)
        parts = content["parts"]
        if not isinstance(parts, list):
            raise LLMProviderError(LLMErrorCode.MALFORMED_RESPONSE)
        text = "".join(
            part["text"]
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
        if not text.strip():
            raise LLMProviderError(LLMErrorCode.EMPTY_RESPONSE)
        usage = data.get("usageMetadata")
        normalized_usage = LLMUsage()
        if isinstance(usage, dict):
            normalized_usage = LLMUsage(
                prompt_tokens=nonnegative_int(usage.get("promptTokenCount")),
                completion_tokens=nonnegative_int(usage.get("candidatesTokenCount")),
                total_tokens=nonnegative_int(usage.get("totalTokenCount")),
            )
        finish_reason = candidate.get("finishReason")
        response_id = data.get("responseId")
        reported_model = data.get("modelVersion")
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

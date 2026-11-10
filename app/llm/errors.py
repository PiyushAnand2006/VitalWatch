"""Safe provider failures; messages never include prompts, completions, or keys."""

from __future__ import annotations

from app.schemas.providers import LLMErrorCode


class LLMProviderError(RuntimeError):
    """A normalized, log-safe provider failure."""

    def __init__(
        self,
        code: LLMErrorCode,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(f"external model request failed: {code.value}")
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class LLMConfigurationError(ValueError):
    """A slot or provider cannot be constructed from the supplied configuration."""

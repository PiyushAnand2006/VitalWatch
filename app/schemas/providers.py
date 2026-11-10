"""Normalized external model contracts, slot configuration, and safe status views."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.privacy import DispatchOutcome, ExternalDestination, PreparedPayload


class ProviderKind(StrEnum):
    """Supported external provider families."""

    NVIDIA = "nvidia"
    GOOGLE = "google"
    LLAMA = "llama"


class ModelSlot(StrEnum):
    """Logical model roles; providers and models are configured explicitly."""

    REASONER_A = "reasoner_a"
    REASONER_B = "reasoner_b"
    REASONER_C = "reasoner_c"
    FINAL_ADJUDICATOR = "final_adjudicator"
    FALLBACK = "fallback"


class LLMErrorCode(StrEnum):
    """Safe provider error codes that never carry a response body or secret."""

    AUTHENTICATION_FAILED = "authentication_failed"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    NETWORK_ERROR = "network_error"
    BAD_REQUEST = "bad_request"
    MALFORMED_RESPONSE = "malformed_response"
    EMPTY_RESPONSE = "empty_response"
    BLOCKED_RESPONSE = "blocked_response"
    SLOT_DISABLED = "slot_disabled"
    EXTERNAL_DISABLED = "external_disabled"
    APPROVAL_REQUIRED = "approval_required"


class RoutingStatus(StrEnum):
    """Terminal state of one routed model call."""

    COMPLETED = "completed"
    BLOCKED = "blocked"
    DISABLED = "disabled"
    FAILED = "failed"


class LLMMessage(BaseModel):
    """One normalized chat message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user"]
    content: str = Field(min_length=1, max_length=200_000)


class LLMRequest(BaseModel):
    """Provider-independent request built only from an approved prepared payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1, max_length=128)
    prompt_id: str = Field(min_length=1, max_length=64)
    prompt_version: str = Field(min_length=1, max_length=32)
    payload: PreparedPayload
    messages: tuple[LLMMessage, ...] = Field(min_length=2, max_length=16)
    max_output_tokens: int = Field(ge=1, le=8192)
    temperature: float = Field(ge=0.0, le=2.0)

    @model_validator(mode="after")
    def require_system_then_user(self) -> LLMRequest:
        """Keep the instruction/data separation unambiguous for every adapter."""

        if self.messages[0].role != "system" or self.messages[1].role != "user":
            raise ValueError("messages must start with a system message then a user message")
        return self


class LLMUsage(BaseModel):
    """Normalized token usage when the host reports it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class LLMResponse(BaseModel):
    """Provider-independent completion with no raw request or secret metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str = Field(min_length=1, max_length=128)
    provider: ProviderKind
    model: str = Field(min_length=1, max_length=128)
    text: str = Field(min_length=1, max_length=200_000)
    finish_reason: str | None = Field(default=None, max_length=64)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    response_id: str | None = Field(default=None, max_length=256)
    reported_model: str | None = Field(default=None, max_length=256)
    latency_ms: int = Field(ge=0)


class RoutingResult(BaseModel):
    """Result of a routed call, including whether an approved fallback was used."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: RoutingStatus
    slot: ModelSlot
    destination: ExternalDestination | None = None
    dispatch_outcome: DispatchOutcome | None = None
    response: LLMResponse | None = None
    error_code: LLMErrorCode | None = None
    fallback_used: bool = False

    @model_validator(mode="after")
    def enforce_result_consistency(self) -> RoutingResult:
        """Prevent a failed or blocked call from carrying completion text."""

        if self.status is RoutingStatus.COMPLETED:
            if self.response is None or self.dispatch_outcome is not DispatchOutcome.DISPATCHED:
                raise ValueError("completed routes require a dispatched response")
        elif self.response is not None:
            raise ValueError("only completed routes may expose a response")
        if self.status is RoutingStatus.FAILED and self.error_code is None:
            raise ValueError("failed routes require a safe error code")
        return self


class ProviderStatus(BaseModel):
    """Credential/configuration state without values or secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ProviderKind
    configured: bool
    api_key_configured: bool
    slot_count: int = Field(ge=0)


class SlotStatus(BaseModel):
    """One slot's configuration and whether it can currently be called."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot: ModelSlot
    configured: bool
    enabled: bool
    provider: ProviderKind | None = None
    model: str | None = None
    endpoint: str | None = None


class ProvidersResponse(BaseModel):
    """Operational model configuration; never includes keys, headers, or prompts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    external_llm_enabled: bool
    local_only: bool
    fallback_enabled: bool
    fallback_requires_fresh_approval: Literal[True] = True
    prompt_version: str
    providers: list[ProviderStatus]
    slots: list[SlotStatus]

"""Resolve typed settings into approval destinations, adapters, and safe status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict
from urllib.parse import quote

from app.core.config import Settings
from app.llm.base import LLMProvider
from app.llm.google import GoogleProvider
from app.llm.openai_compatible import LlamaCompatibleProvider, NvidiaProvider
from app.llm.prompts import PROMPT_VERSION
from app.llm.rate_limit import AsyncRateLimiter
from app.schemas.privacy import DispatchPurpose, ExternalDestination
from app.schemas.providers import (
    ModelSlot,
    ProviderKind,
    ProvidersResponse,
    ProviderStatus,
    SlotStatus,
)


@dataclass(frozen=True)
class SlotConfig:
    """One configured slot and the exact destination that must be approved."""

    slot: ModelSlot
    provider: ProviderKind
    model: str
    endpoint: str
    purpose: DispatchPurpose

    def destination(self, purpose: DispatchPurpose | None = None) -> ExternalDestination:
        """Build the approval-bound destination for this slot.

        Debate nodes call one slot for several purposes (reasoning, critique,
        revision); each purpose is a distinct approval destination.
        """

        return ExternalDestination(
            provider=self.provider.value,
            endpoint=self.endpoint,
            model=self.model,
            purpose=purpose or self.purpose,
        )


@dataclass
class ProviderRegistry:
    """Per-slot adapters sharing a per-provider rate limiter."""

    providers: dict[ModelSlot, LLMProvider]

    async def aclose(self) -> None:
        """Release every adapter-owned HTTP client."""

        for provider in self.providers.values():
            await provider.aclose()


@dataclass(frozen=True)
class _SlotSpec:
    """One slot's settings prefix and its default approval purpose."""

    prefix: str
    purpose: DispatchPurpose


_SLOT_SPECS: dict[ModelSlot, _SlotSpec] = {
    ModelSlot.REASONER_A: _SlotSpec("reasoner_a", DispatchPurpose.REASONING),
    ModelSlot.REASONER_B: _SlotSpec("reasoner_b", DispatchPurpose.REASONING),
    ModelSlot.REASONER_C: _SlotSpec("reasoner_c", DispatchPurpose.REASONING),
    ModelSlot.FINAL_ADJUDICATOR: _SlotSpec("adjudicator", DispatchPurpose.ADJUDICATION),
    ModelSlot.FALLBACK: _SlotSpec("fallback", DispatchPurpose.FALLBACK),
}


def _slot_values(
    settings: Settings, slot: ModelSlot
) -> tuple[ProviderKind | None, str | None, str | None]:
    prefix = _SLOT_SPECS[slot].prefix
    return (
        getattr(settings, f"llm_{prefix}_provider"),
        getattr(settings, f"llm_{prefix}_model"),
        getattr(settings, f"llm_{prefix}_endpoint"),
    )


def _provider_endpoint(
    settings: Settings, provider: ProviderKind, model: str, override: str | None
) -> str:
    if override is not None:
        return override
    if provider is ProviderKind.NVIDIA:
        return settings.nvidia_chat_endpoint
    if provider is ProviderKind.GOOGLE:
        model_id = model.removeprefix("models/")
        return (
            f"{settings.google_api_base_url.rstrip('/')}"
            f"/models/{quote(model_id, safe='')}:generateContent"
        )
    if settings.llama_base_url is None:
        raise ValueError("Llama provider requires LLAMA_BASE_URL")
    return (
        f"{settings.llama_base_url.rstrip('/')}"
        f"/{settings.llama_chat_route.lstrip('/')}"
    )


def resolve_slots(settings: Settings) -> dict[ModelSlot, SlotConfig]:
    """Return only fully configured slots; all others remain unavailable."""

    slots: dict[ModelSlot, SlotConfig] = {}
    for slot, spec in _SLOT_SPECS.items():
        provider, model, endpoint = _slot_values(settings, slot)
        if provider is None or model is None:
            continue
        slots[slot] = SlotConfig(
            slot=slot,
            provider=provider,
            model=model,
            endpoint=_provider_endpoint(settings, provider, model, endpoint),
            purpose=spec.purpose,
        )
    return slots


def _api_key(settings: Settings, provider: ProviderKind) -> str | None:
    value = {
        ProviderKind.NVIDIA: settings.nvidia_api_key,
        ProviderKind.GOOGLE: settings.google_api_key,
        ProviderKind.LLAMA: settings.llama_api_key,
    }[provider]
    return value.get_secret_value() if value is not None else None


class _ProviderOptions(TypedDict):
    model: str
    endpoint: str
    api_key: str | None
    timeout_seconds: float
    max_retries: int
    retry_base_delay_seconds: float
    rate_limiter: AsyncRateLimiter


def build_provider_registry(
    settings: Settings, slots: dict[ModelSlot, SlotConfig]
) -> ProviderRegistry:
    """Build one adapter per enabled-or-configured slot, sharing rate limits."""

    limiters = {
        provider: AsyncRateLimiter(settings.llm_rate_limit_per_minute)
        for provider in ProviderKind
    }
    providers: dict[ModelSlot, LLMProvider] = {}
    for slot, config in slots.items():
        common = _ProviderOptions(
            model=config.model,
            endpoint=config.endpoint,
            api_key=_api_key(settings, config.provider),
            timeout_seconds=settings.llm_request_timeout_seconds,
            max_retries=settings.llm_max_retries,
            retry_base_delay_seconds=settings.llm_retry_base_delay_seconds,
            rate_limiter=limiters[config.provider],
        )
        if config.provider is ProviderKind.NVIDIA:
            providers[slot] = NvidiaProvider(**common)
        elif config.provider is ProviderKind.GOOGLE:
            providers[slot] = GoogleProvider(**common)
        else:
            providers[slot] = LlamaCompatibleProvider(
                auth_header=settings.llama_auth_header,
                auth_scheme=settings.llama_auth_scheme,
                **common,
            )
    return ProviderRegistry(providers=providers)


def describe_providers(
    settings: Settings, slots: dict[ModelSlot, SlotConfig]
) -> ProvidersResponse:
    """Build a route response containing no keys, headers, or prompts."""

    provider_statuses = []
    for provider in ProviderKind:
        count = sum(1 for config in slots.values() if config.provider is provider)
        if provider is ProviderKind.NVIDIA:
            configured = settings.nvidia_api_key is not None
        elif provider is ProviderKind.GOOGLE:
            configured = settings.google_api_key is not None
        else:
            configured = settings.llama_base_url is not None
        provider_statuses.append(
            ProviderStatus(
                provider=provider,
                configured=configured,
                api_key_configured=_api_key(settings, provider) is not None,
                slot_count=count,
            )
        )

    callable_now = settings.external_llm_enabled and not settings.privacy_local_only
    slot_statuses = [
        SlotStatus(
            slot=slot,
            configured=slot in slots,
            enabled=slot in slots and callable_now,
            provider=slots[slot].provider if slot in slots else None,
            model=slots[slot].model if slot in slots else None,
            endpoint=slots[slot].endpoint if slot in slots else None,
        )
        for slot in _SLOT_SPECS
    ]
    return ProvidersResponse(
        external_llm_enabled=settings.external_llm_enabled,
        local_only=settings.privacy_local_only,
        fallback_enabled=settings.llm_fallback_enabled,
        prompt_version=PROMPT_VERSION,
        providers=provider_statuses,
        slots=slot_statuses,
    )

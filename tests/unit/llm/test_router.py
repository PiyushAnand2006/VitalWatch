"""Approval-bound routing, disabled slots, fallback re-approval, and redaction tests."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from app.llm.config import SlotConfig
from app.llm.errors import LLMProviderError
from app.llm.router import ProviderRouter
from app.privacy.dispatch import ApprovedDispatcher
from app.privacy.gate import PrivacyGate
from app.schemas.evidence import EvidencePackage
from app.schemas.privacy import DispatchPurpose
from app.schemas.providers import LLMErrorCode, ModelSlot, ProviderKind, RoutingStatus
from tests.privacy_fixtures import MemoryApprovalStore, StaticAnalyzer, make_evidence
from tests.unit.llm.conftest import FakeProvider, make_request, make_response

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
PRIMARY_ENDPOINT = (
    "https://generativelanguage.example.invalid/v1/models/synthetic-reasoner:generateContent"
)
FALLBACK_ENDPOINT = "https://integrate.api.example.invalid/v1/chat/completions"


def reasoner_config() -> SlotConfig:
    return SlotConfig(
        slot=ModelSlot.REASONER_A,
        provider=ProviderKind.GOOGLE,
        model="synthetic-reasoner",
        endpoint=PRIMARY_ENDPOINT,
        purpose=DispatchPurpose.REASONING,
    )


def fallback_config() -> SlotConfig:
    return SlotConfig(
        slot=ModelSlot.FALLBACK,
        provider=ProviderKind.NVIDIA,
        model="synthetic-fallback",
        endpoint=FALLBACK_ENDPOINT,
        purpose=DispatchPurpose.FALLBACK,
    )


def make_router(
    slots: dict[ModelSlot, SlotConfig],
    providers: dict[ModelSlot, FakeProvider],
    store: MemoryApprovalStore,
    *,
    external_enabled: bool = True,
    local_only: bool = False,
    fallback_enabled: bool = False,
) -> ProviderRouter:
    return ProviderRouter(
        slots=slots,
        providers=providers,  # type: ignore[arg-type]
        dispatcher=ApprovedDispatcher(approvals=store, local_only=local_only, clock=lambda: NOW),
        external_enabled=external_enabled,
        local_only=local_only,
        fallback_enabled=fallback_enabled,
    )


def make_gate(store: MemoryApprovalStore) -> PrivacyGate:
    return PrivacyGate(
        analyzer=StaticAnalyzer(),
        approvals=store,
        local_only=False,
        clock=lambda: NOW,
        approval_ttl_seconds=300,
    )


async def approve_for(
    gate: PrivacyGate, evidence: EvidencePackage, destination: object
) -> object:
    decision = await gate.prepare(
        evidence=evidence,
        question="Interpret the potassium result.",
        destination=destination,  # type: ignore[arg-type]
        request_id="req-router-001",
    )
    assert decision.payload is not None
    assert decision.status.value == "approved"
    return decision


async def approved_setup(
    *,
    fallback_enabled: bool = False,
    provider_error: LLMProviderError | None = None,
) -> tuple[ProviderRouter, MemoryApprovalStore, FakeProvider, FakeProvider, object, object]:
    """Mint real destination-bound approvals and wire a router around fakes."""

    store = MemoryApprovalStore()
    gate = make_gate(store)
    evidence = make_evidence()

    primary = reasoner_config()
    primary_decision = await approve_for(gate, evidence, primary.destination())

    fallback_decision: object | None = None
    if fallback_enabled:
        fallback_decision = await approve_for(gate, evidence, fallback_config().destination())

    primary_provider = FakeProvider(
        kind=ProviderKind.GOOGLE,
        model=primary.model,
        endpoint=primary.endpoint,
        response=make_response(kind=ProviderKind.GOOGLE, model=primary.model),
        error=provider_error,
    )
    fallback_provider = FakeProvider(
        kind=ProviderKind.NVIDIA,
        model="synthetic-fallback",
        endpoint=FALLBACK_ENDPOINT,
        response=make_response(kind=ProviderKind.NVIDIA, model="synthetic-fallback"),
    )
    slots = {ModelSlot.REASONER_A: primary}
    providers = {ModelSlot.REASONER_A: primary_provider}
    if fallback_enabled:
        slots[ModelSlot.FALLBACK] = fallback_config()
        providers[ModelSlot.FALLBACK] = fallback_provider
    router = make_router(
        slots=slots,
        providers=providers,
        store=store,
        fallback_enabled=fallback_enabled,
    )
    return router, store, primary_provider, fallback_provider, primary_decision, fallback_decision


async def test_approved_destination_completes_the_route() -> None:
    router, store, primary, fallback, decision, _ = await approved_setup()

    result = await router.generate(
        slot=ModelSlot.REASONER_A,
        request=make_request(payload=decision.payload),  # type: ignore[arg-type]
        approval_id=decision.approval_id,  # type: ignore[attr-defined]
    )

    assert result.status is RoutingStatus.COMPLETED
    assert result.response is not None
    assert result.response.text == "Synthetic explanation."
    assert result.fallback_used is False
    assert primary.calls == 1
    assert fallback.calls == 0
    assert decision.approval_id in store.consumed  # type: ignore[attr-defined]


async def test_unconfigured_slot_is_disabled_without_provider_calls() -> None:
    router, _store, primary, _fallback, _decision, _unused = await approved_setup()

    result = await router.generate(
        slot=ModelSlot.REASONER_B,
        request=make_request(),
        approval_id=uuid4(),
    )

    assert result.status is RoutingStatus.DISABLED
    assert result.error_code is LLMErrorCode.SLOT_DISABLED
    assert primary.calls == 0


async def test_external_disabled_blocks_every_slot() -> None:
    store = MemoryApprovalStore()
    primary = reasoner_config()
    provider = FakeProvider(kind=ProviderKind.GOOGLE, model="m", endpoint=primary.endpoint)
    router = make_router(
        slots={ModelSlot.REASONER_A: primary},
        providers={ModelSlot.REASONER_A: provider},
        store=store,
        external_enabled=False,
    )

    result = await router.generate(
        slot=ModelSlot.REASONER_A, request=make_request(), approval_id=uuid4()
    )

    assert result.status is RoutingStatus.DISABLED
    assert result.error_code is LLMErrorCode.EXTERNAL_DISABLED
    assert provider.calls == 0


async def test_local_only_mode_blocks_before_the_store_is_touched() -> None:
    store = MemoryApprovalStore()
    store.get_error = RuntimeError("store must not be reached")
    primary = reasoner_config()
    provider = FakeProvider(kind=ProviderKind.GOOGLE, model="m", endpoint=primary.endpoint)
    router = make_router(
        slots={ModelSlot.REASONER_A: primary},
        providers={ModelSlot.REASONER_A: provider},
        store=store,
        local_only=True,
    )

    result = await router.generate(
        slot=ModelSlot.REASONER_A, request=make_request(), approval_id=uuid4()
    )

    # Short-circuited as disabled before any dispatch: a reached dispatcher would
    # have returned BLOCKED_STORE_FAILURE instead of DISABLED.
    assert result.status is RoutingStatus.DISABLED
    assert result.error_code is LLMErrorCode.EXTERNAL_DISABLED
    assert provider.calls == 0


async def test_wrong_destination_approval_is_blocked() -> None:
    router, _store, primary, _fallback, decision, _unused = await approved_setup()

    # The approval binds reasoner_a's destination; calling the adjudicator slot with it
    # would use a different destination, which must never dispatch.
    adjudicator = SlotConfig(
        slot=ModelSlot.FINAL_ADJUDICATOR,
        provider=ProviderKind.GOOGLE,
        model="synthetic-reasoner",
        endpoint=PRIMARY_ENDPOINT,
        purpose=DispatchPurpose.ADJUDICATION,
    )
    full_router = make_router(
        slots={
            ModelSlot.REASONER_A: reasoner_config(),
            ModelSlot.FINAL_ADJUDICATOR: adjudicator,
        },
        providers={
            ModelSlot.REASONER_A: primary,
            ModelSlot.FINAL_ADJUDICATOR: FakeProvider(
                kind=ProviderKind.GOOGLE,
                model="synthetic-reasoner",
                endpoint=PRIMARY_ENDPOINT,
                response=make_response(),
            ),
        },
        store=_store,
    )

    result = await full_router.generate(
        slot=ModelSlot.FINAL_ADJUDICATOR,
        request=make_request(payload=decision.payload),  # type: ignore[arg-type]
        approval_id=decision.approval_id,  # type: ignore[attr-defined]
    )

    assert result.status is RoutingStatus.BLOCKED
    assert result.error_code is LLMErrorCode.APPROVAL_REQUIRED
    assert primary.calls == 0


async def test_mutated_payload_fails_the_digest_check() -> None:
    router, _store, primary, _fallback, decision, _unused = await approved_setup()
    mutated = decision.payload.model_copy(update={"question": "A different question"})  # type: ignore[attr-defined]

    result = await router.generate(
        slot=ModelSlot.REASONER_A,
        request=make_request(payload=mutated),
        approval_id=decision.approval_id,  # type: ignore[attr-defined]
    )

    assert result.status is RoutingStatus.BLOCKED
    assert result.error_code is LLMErrorCode.APPROVAL_REQUIRED
    assert primary.calls == 0


async def test_consumed_approval_cannot_dispatch_twice() -> None:
    router, _store, primary, _fallback, decision, _unused = await approved_setup()

    first = await router.generate(
        slot=ModelSlot.REASONER_A,
        request=make_request(payload=decision.payload),  # type: ignore[arg-type]
        approval_id=decision.approval_id,  # type: ignore[attr-defined]
    )
    second = await router.generate(
        slot=ModelSlot.REASONER_A,
        request=make_request(payload=decision.payload),  # type: ignore[arg-type]
        approval_id=decision.approval_id,  # type: ignore[attr-defined]
    )

    assert first.status is RoutingStatus.COMPLETED
    assert second.status is RoutingStatus.BLOCKED
    assert second.error_code is LLMErrorCode.APPROVAL_REQUIRED
    assert primary.calls == 1


async def test_fallback_without_fresh_approval_never_switches_destination() -> None:
    router, _store, primary, fallback, decision, _unused = await approved_setup(
        fallback_enabled=True,
        provider_error=LLMProviderError(LLMErrorCode.UNAVAILABLE, retryable=True),
    )

    # Reusing the consumed primary approval for the fallback destination must be blocked.
    result = await router.generate(
        slot=ModelSlot.REASONER_A,
        request=make_request(payload=decision.payload),  # type: ignore[arg-type]
        approval_id=decision.approval_id,  # type: ignore[attr-defined]
        fallback_approval_id=decision.approval_id,  # type: ignore[attr-defined]
    )

    assert result.status is RoutingStatus.BLOCKED
    assert result.error_code is LLMErrorCode.APPROVAL_REQUIRED
    assert result.fallback_used is True
    assert primary.calls == 1
    assert fallback.calls == 0


async def test_fallback_uses_a_separately_approved_destination() -> None:
    router, _store, primary, fallback, decision, fallback_decision = await approved_setup(
        fallback_enabled=True,
        provider_error=LLMProviderError(LLMErrorCode.UNAVAILABLE, retryable=True),
    )

    result = await router.generate(
        slot=ModelSlot.REASONER_A,
        request=make_request(payload=decision.payload),  # type: ignore[arg-type]
        approval_id=decision.approval_id,  # type: ignore[attr-defined]
        fallback_approval_id=fallback_decision.approval_id,  # type: ignore[attr-defined]
    )

    assert result.status is RoutingStatus.COMPLETED
    assert result.fallback_used is True
    assert result.response is not None
    assert result.response.provider is ProviderKind.NVIDIA
    assert primary.calls == 1
    assert fallback.calls == 1


async def test_fallback_without_approval_configured_returns_primary_failure() -> None:
    router, _store, primary, fallback, decision, _unused = await approved_setup(
        fallback_enabled=True,
        provider_error=LLMProviderError(LLMErrorCode.UNAVAILABLE, retryable=True),
    )

    result = await router.generate(
        slot=ModelSlot.REASONER_A,
        request=make_request(payload=decision.payload),  # type: ignore[arg-type]
        approval_id=decision.approval_id,  # type: ignore[attr-defined]
        fallback_approval_id=None,
    )

    assert result.status is RoutingStatus.FAILED
    assert result.error_code is LLMErrorCode.UNAVAILABLE
    assert result.fallback_used is False
    assert fallback.calls == 0


async def test_fallback_disabled_returns_primary_failure() -> None:
    router, _store, primary, fallback, decision, _unused = await approved_setup(
        fallback_enabled=False,
        provider_error=LLMProviderError(LLMErrorCode.UNAVAILABLE, retryable=True),
    )

    result = await router.generate(
        slot=ModelSlot.REASONER_A,
        request=make_request(payload=decision.payload),  # type: ignore[arg-type]
        approval_id=decision.approval_id,  # type: ignore[attr-defined]
    )

    assert result.status is RoutingStatus.FAILED
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.parametrize(
    "code",
    [
        LLMErrorCode.AUTHENTICATION_FAILED,
        LLMErrorCode.BAD_REQUEST,
        LLMErrorCode.MALFORMED_RESPONSE,
        LLMErrorCode.EMPTY_RESPONSE,
        LLMErrorCode.BLOCKED_RESPONSE,
    ],
)
async def test_non_transient_errors_never_trigger_fallback(code: LLMErrorCode) -> None:
    router, _store, primary, fallback, decision, fallback_decision = await approved_setup(
        fallback_enabled=True,
        provider_error=LLMProviderError(code),
    )

    result = await router.generate(
        slot=ModelSlot.REASONER_A,
        request=make_request(payload=decision.payload),  # type: ignore[arg-type]
        approval_id=decision.approval_id,  # type: ignore[attr-defined]
        fallback_approval_id=fallback_decision.approval_id,  # type: ignore[attr-defined]
    )

    assert result.status is RoutingStatus.FAILED
    assert result.error_code is code
    assert fallback.calls == 0


async def test_unconfigured_fallback_slot_returns_primary_failure() -> None:
    router, _store, primary, _fallback, decision, _unused = await approved_setup(
        provider_error=LLMProviderError(LLMErrorCode.UNAVAILABLE, retryable=True)
    )

    result = await router.generate(
        slot=ModelSlot.REASONER_A,
        request=make_request(payload=decision.payload),  # type: ignore[arg-type]
        approval_id=decision.approval_id,  # type: ignore[attr-defined]
        fallback_approval_id=uuid4(),
    )

    assert result.status is RoutingStatus.FAILED
    assert result.error_code is LLMErrorCode.UNAVAILABLE
    assert primary.calls == 1


async def test_provider_failures_log_no_prompts_or_secrets(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router, _store, _primary, _fallback, decision, _unused = await approved_setup(
        provider_error=LLMProviderError(LLMErrorCode.TIMEOUT, retryable=True)
    )
    synthetic_question = "Interpret the potassium result."

    with caplog.at_level(logging.DEBUG):
        result = await router.generate(
            slot=ModelSlot.REASONER_A,
            request=make_request(payload=decision.payload),  # type: ignore[arg-type]
            approval_id=decision.approval_id,  # type: ignore[attr-defined]
        )

    assert result.status is RoutingStatus.FAILED
    assert synthetic_question not in caplog.text
    assert decision.payload.question not in caplog.text  # type: ignore[attr-defined]
    assert "Synthetic explanation." not in caplog.text

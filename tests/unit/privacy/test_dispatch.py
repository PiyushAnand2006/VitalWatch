"""Single-use dispatch guard tests."""

from datetime import UTC, datetime, timedelta

from app.privacy.dispatch import ApprovedDispatcher
from app.privacy.gate import PrivacyGate
from app.schemas.privacy import (
    DispatchOutcome,
    ExternalDestination,
    PrivacyStatus,
)
from tests.privacy_fixtures import MemoryApprovalStore, StaticAnalyzer, make_evidence

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
DESTINATION = ExternalDestination(
    provider="google",
    endpoint="https://generativelanguage.example.invalid/v1/models/x:generateContent",
    model="synthetic-reasoner",
    purpose="reasoning",
)
OTHER_DESTINATION = DESTINATION.model_copy(update={"model": "other-reasoner"})


class SpyTransport:
    def __init__(self) -> None:
        self.calls = 0
        self.payloads: list[str] = []

    async def __call__(self, payload, destination) -> str:
        self.calls += 1
        self.payloads.append(payload.question)
        return "provider-response"


async def approved(gate: PrivacyGate, store: MemoryApprovalStore):
    decision = await gate.prepare(
        evidence=make_evidence(),
        question="Interpret the potassium result.",
        destination=DESTINATION,
        request_id="req-dispatch",
    )
    assert decision.status is PrivacyStatus.APPROVED
    assert decision.payload is not None
    return decision, store.records[decision.approval_id]


def make_gate(store: MemoryApprovalStore) -> PrivacyGate:
    return PrivacyGate(
        analyzer=StaticAnalyzer(),
        approvals=store,
        local_only=False,
        clock=lambda: NOW,
        approval_ttl_seconds=300,
    )


async def test_approved_payload_is_dispatched_once() -> None:
    store = MemoryApprovalStore()
    gate = make_gate(store)
    decision, record = await approved(gate, store)
    dispatcher = ApprovedDispatcher(approvals=store, local_only=False, clock=lambda: NOW)
    transport = SpyTransport()

    first = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=decision.payload,
        destination=DESTINATION,
        transport=transport,
    )
    second = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=decision.payload,
        destination=DESTINATION,
        transport=transport,
    )

    assert first.outcome is DispatchOutcome.DISPATCHED
    assert first.value == "provider-response"
    assert second.outcome is DispatchOutcome.BLOCKED_CONSUMED
    assert transport.calls == 1
    assert store.consumed == {record.approval_id}


async def test_mutated_payload_invalidates_the_approval() -> None:
    store = MemoryApprovalStore()
    gate = make_gate(store)
    decision, _ = await approved(gate, store)
    dispatcher = ApprovedDispatcher(approvals=store, local_only=False, clock=lambda: NOW)
    transport = SpyTransport()
    mutated = decision.payload.model_copy(update={"question": "A different question"})

    result = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=mutated,
        destination=DESTINATION,
        transport=transport,
    )

    assert result.outcome is DispatchOutcome.BLOCKED_DIGEST_MISMATCH
    assert transport.calls == 0


async def test_wrong_destination_is_blocked() -> None:
    store = MemoryApprovalStore()
    gate = make_gate(store)
    decision, _ = await approved(gate, store)
    dispatcher = ApprovedDispatcher(approvals=store, local_only=False, clock=lambda: NOW)
    transport = SpyTransport()

    result = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=decision.payload,
        destination=OTHER_DESTINATION,
        transport=transport,
    )

    assert result.outcome is DispatchOutcome.BLOCKED_DESTINATION_MISMATCH
    assert transport.calls == 0


async def test_local_only_blocks_before_touching_the_store() -> None:
    store = MemoryApprovalStore()
    store.get_error = RuntimeError("store must not be reached")
    gate = make_gate(store)
    decision, _ = await approved(gate, store)
    dispatcher = ApprovedDispatcher(approvals=store, local_only=True, clock=lambda: NOW)
    transport = SpyTransport()

    result = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=decision.payload,
        destination=DESTINATION,
        transport=transport,
    )

    assert result.outcome is DispatchOutcome.BLOCKED_LOCAL_ONLY
    assert transport.calls == 0


async def test_expired_approval_is_blocked_without_consuming() -> None:
    store = MemoryApprovalStore()
    gate = make_gate(store)
    decision, _ = await approved(gate, store)
    too_late = NOW + timedelta(seconds=301)
    dispatcher = ApprovedDispatcher(
        approvals=store, local_only=False, clock=lambda: too_late
    )
    transport = SpyTransport()

    result = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=decision.payload,
        destination=DESTINATION,
        transport=transport,
    )

    assert result.outcome is DispatchOutcome.BLOCKED_EXPIRED
    assert transport.calls == 0
    assert decision.approval_id not in store.consumed


async def test_non_approved_record_cannot_dispatch() -> None:
    store = MemoryApprovalStore()
    gate = make_gate(store)
    decision, record = await approved(gate, store)
    store.records[record.approval_id] = record.model_copy(
        update={"status": PrivacyStatus.NEEDS_REVIEW}
    )
    dispatcher = ApprovedDispatcher(approvals=store, local_only=False, clock=lambda: NOW)
    transport = SpyTransport()

    result = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=decision.payload,
        destination=DESTINATION,
        transport=transport,
    )

    assert result.outcome is DispatchOutcome.BLOCKED_NOT_APPROVED
    assert transport.calls == 0


async def test_store_failures_fail_closed() -> None:
    store = MemoryApprovalStore()
    gate = make_gate(store)
    decision, _ = await approved(gate, store)
    store.get_error = RuntimeError("database unavailable")
    dispatcher = ApprovedDispatcher(approvals=store, local_only=False, clock=lambda: NOW)
    transport = SpyTransport()

    get_result = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=decision.payload,
        destination=DESTINATION,
        transport=transport,
    )

    store.get_error = None
    store.consume_error = RuntimeError("database unavailable")
    consume_result = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=decision.payload,
        destination=DESTINATION,
        transport=transport,
    )

    assert get_result.outcome is DispatchOutcome.BLOCKED_STORE_FAILURE
    assert consume_result.outcome is DispatchOutcome.BLOCKED_STORE_FAILURE
    assert transport.calls == 0


async def test_fallback_requires_a_fresh_approval() -> None:
    store = MemoryApprovalStore()
    gate = make_gate(store)
    decision, _ = await approved(gate, store)
    dispatcher = ApprovedDispatcher(approvals=store, local_only=False, clock=lambda: NOW)
    transport = SpyTransport()
    fallback_destination = DESTINATION.model_copy(
        update={"provider": "fallback-provider"}
    )

    first = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=decision.payload,
        destination=DESTINATION,
        transport=transport,
    )
    fallback = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=decision.payload,
        destination=fallback_destination,
        transport=transport,
    )

    assert first.outcome is DispatchOutcome.DISPATCHED
    assert fallback.outcome in {
        DispatchOutcome.BLOCKED_CONSUMED,
        DispatchOutcome.BLOCKED_DESTINATION_MISMATCH,
    }
    assert transport.calls == 1

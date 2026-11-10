"""Stage 6 privacy gate against real Presidio, real PostgreSQL, and the real migration."""

from datetime import UTC, datetime
from decimal import Decimal

import psycopg
import pytest

from app.core.database import DatabaseSessionManager
from app.privacy.dispatch import ApprovedDispatcher
from app.privacy.engine import PresidioPiiAnalyzer
from app.privacy.gate import PrivacyGate
from app.repositories.privacy import PrivacyApprovalRepository
from app.schemas.privacy import (
    DispatchOutcome,
    DispatchPurpose,
    ExternalDestination,
    PrivacyStatus,
)
from tests.privacy_fixtures import PATIENT_ID, make_evidence, make_observation

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
DESTINATION = ExternalDestination(
    provider="nvidia",
    endpoint="https://api.example.invalid/v1/chat/completions",
    model="synthetic-reasoner",
    purpose=DispatchPurpose.REASONING,
)
EMAIL_QUESTION = "What does the potassium result mean? Contact john@example.com"


class SpyTransport:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, payload, destination) -> str:
        self.calls += 1
        return "provider-response"


def gate_for(
    sessions: DatabaseSessionManager,
    *,
    local_only: bool = False,
) -> tuple[PrivacyGate, PrivacyApprovalRepository]:
    repository = PrivacyApprovalRepository(sessions.sessionmaker)
    gate = PrivacyGate(
        analyzer=PresidioPiiAnalyzer(),
        approvals=repository,
        local_only=local_only,
        clock=lambda: NOW,
        approval_ttl_seconds=300,
    )
    return gate, repository


def row_json(
    connection: psycopg.Connection[tuple[object, ...]], approval_id: object
) -> str:
    row = connection.execute(
        "select row_to_json(t)::text from privacy_approvals t where id = %s",
        (approval_id,),
    ).fetchone()
    assert row is not None
    return row[0]


async def test_migration_creates_the_privacy_approval_table(
    postgres_connection: psycopg.Connection[tuple[object, ...]],
) -> None:
    columns = postgres_connection.execute(
        "select column_name from information_schema.columns "
        "where table_name = 'privacy_approvals'"
    ).fetchall()

    names = {row[0] for row in columns}
    assert {
        "id",
        "request_id",
        "status",
        "destination_digest",
        "payload_sha256",
        "patient_ref",
        "actor_ref",
        "review_flags",
        "entity_counts",
        "consumed_at",
    } <= names


async def test_approved_decision_persists_only_hashes_and_counts(
    sessions: DatabaseSessionManager,
    postgres_connection: psycopg.Connection[tuple[object, ...]],
) -> None:
    gate, repository = gate_for(sessions)

    decision = await gate.prepare(
        evidence=make_evidence(patient_id=PATIENT_ID),
        question=EMAIL_QUESTION,
        destination=DESTINATION,
        request_id="req-integration-approved",
        actor_ref="clinician@example.com",
    )

    assert decision.status is PrivacyStatus.APPROVED
    assert decision.payload is not None
    stored = await repository.get(decision.approval_id)
    assert stored is not None
    assert stored.payload_sha256 == decision.payload_sha256
    assert stored.patient_ref != PATIENT_ID
    assert stored.actor_ref != "clinician@example.com"

    serialized_row = row_json(postgres_connection, decision.approval_id)
    for raw in (
        PATIENT_ID,
        "John Smith",
        "john@example.com",
        "clinician@example.com",
        "chunk-1",
        "a" * 64,
    ):
        assert raw not in serialized_row
    assert "approved" in serialized_row


async def test_review_decision_is_audited_without_a_payload(
    sessions: DatabaseSessionManager,
    postgres_connection: psycopg.Connection[tuple[object, ...]],
) -> None:
    gate, repository = gate_for(sessions)
    question = (
        "John Smith, MRN: AB-123456, at Mercy Hospital: "
        "what does the potassium result mean?"
    )

    decision = await gate.prepare(
        evidence=make_evidence(),
        question=question,
        destination=DESTINATION,
        request_id="req-integration-review",
    )

    assert decision.status is PrivacyStatus.NEEDS_REVIEW
    assert decision.payload is None
    stored = await repository.get(decision.approval_id)
    assert stored is not None
    assert stored.status is PrivacyStatus.NEEDS_REVIEW
    assert stored.payload_sha256 is None
    assert stored.review_flags
    serialized_row = row_json(postgres_connection, decision.approval_id)
    for raw in ("John Smith", "AB-123456", "Mercy Hospital", PATIENT_ID):
        assert raw not in serialized_row


async def test_destination_and_digest_guards_run_before_consumption(
    sessions: DatabaseSessionManager,
    postgres_connection: psycopg.Connection[tuple[object, ...]],
) -> None:
    gate, repository = gate_for(sessions)
    decision = await gate.prepare(
        evidence=make_evidence(),
        question=EMAIL_QUESTION,
        destination=DESTINATION,
        request_id="req-integration-guards",
    )
    assert decision.payload is not None
    dispatcher = ApprovedDispatcher(approvals=repository, local_only=False, clock=lambda: NOW)
    transport = SpyTransport()

    mutated = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=decision.payload.model_copy(update={"question": "mutated"}),
        destination=DESTINATION,
        transport=transport,
    )
    wrong_destination = await dispatcher.dispatch(
        approval_id=decision.approval_id,
        payload=decision.payload,
        destination=DESTINATION.model_copy(update={"model": "other-reasoner"}),
        transport=transport,
    )

    assert mutated.outcome is DispatchOutcome.BLOCKED_DIGEST_MISMATCH
    assert wrong_destination.outcome is DispatchOutcome.BLOCKED_DESTINATION_MISMATCH
    assert transport.calls == 0
    unconsumed = postgres_connection.execute(
        "select consumed_at from privacy_approvals where id = %s",
        (decision.approval_id,),
    ).fetchone()
    assert unconsumed is not None and unconsumed[0] is None


async def test_approved_decision_is_consumed_exactly_once(
    sessions: DatabaseSessionManager,
    postgres_connection: psycopg.Connection[tuple[object, ...]],
) -> None:
    gate, repository = gate_for(sessions)
    decision = await gate.prepare(
        evidence=make_evidence(),
        question=EMAIL_QUESTION,
        destination=DESTINATION,
        request_id="req-integration-consume",
    )
    assert decision.payload is not None
    dispatcher = ApprovedDispatcher(approvals=repository, local_only=False, clock=lambda: NOW)
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
    assert second.outcome is DispatchOutcome.BLOCKED_CONSUMED
    assert transport.calls == 1
    consumed = postgres_connection.execute(
        "select consumed_at from privacy_approvals where id = %s",
        (decision.approval_id,),
    ).fetchone()
    assert consumed is not None and consumed[0] is not None


async def test_local_only_is_rejected_and_records_no_payload(
    sessions: DatabaseSessionManager,
    postgres_connection: psycopg.Connection[tuple[object, ...]],
) -> None:
    gate, _ = gate_for(sessions, local_only=True)

    decision = await gate.prepare(
        evidence=make_evidence(),
        question=EMAIL_QUESTION,
        destination=DESTINATION,
        request_id="req-integration-local",
    )

    assert decision.status is PrivacyStatus.REJECTED
    assert decision.reason_codes == ["local_only_mode"]
    assert decision.payload is None
    row = postgres_connection.execute(
        "select status, payload_sha256, reason_codes::text "
        "from privacy_approvals where id = %s",
        (decision.approval_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == "rejected"
    assert row[1] is None
    assert "local_only_mode" in row[2]


async def test_invalid_provenance_is_rejected_before_presidio(
    sessions: DatabaseSessionManager,
    postgres_connection: psycopg.Connection[tuple[object, ...]],
) -> None:
    gate, _ = gate_for(sessions)
    observation = make_observation().model_copy(
        update={"value_decimal": Decimal("9.9")}
    )

    decision = await gate.prepare(
        evidence=make_evidence(observations=[observation]),
        question=EMAIL_QUESTION,
        destination=DESTINATION,
        request_id="req-integration-provenance",
    )

    assert decision.status is PrivacyStatus.REJECTED
    assert "value_mismatch" in decision.reason_codes
    row = postgres_connection.execute(
        "select reason_codes::text from privacy_approvals where id = %s",
        (decision.approval_id,),
    ).fetchone()
    assert row is not None and "value_mismatch" in row[0]

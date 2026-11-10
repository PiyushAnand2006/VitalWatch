"""Minimal audit contract and sink tests."""

import logging
from datetime import datetime
from typing import Any

import pytest
from pydantic import ValidationError

from app.core.audit import AuditService, LoggingAuditSink
from app.schemas.audit import AuditEvent


class CollectingAuditSink:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def record(self, event: AuditEvent) -> None:
        self.events.append(event)


async def test_audit_service_constructs_and_dispatches_valid_event() -> None:
    sink = CollectingAuditSink()
    service = AuditService(sink)

    event = await service.record(
        "configuration.validated",
        "success",
        request_id="request-synthetic-1",
        resource_type="application",
        resource_ref="stage1",
    )

    assert sink.events == [event]
    assert event.event_type == "configuration.validated"
    assert event.occurred_at.tzinfo is not None


def test_audit_event_rejects_arbitrary_payload_fields() -> None:
    payload: dict[str, Any] = {
        "event_type": "configuration.validated",
        "request_id": "request-synthetic-1",
        "outcome": "success",
        "raw_payload": "must not be accepted",
    }

    with pytest.raises(ValidationError):
        AuditEvent.model_validate(payload)


def test_audit_event_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        AuditEvent(
            event_type="configuration.validated",
            request_id="request-synthetic-1",
            outcome="success",
            occurred_at=datetime(2026, 1, 1),
        )


async def test_logging_sink_does_not_emit_actor_or_resource_references(
    caplog: pytest.LogCaptureFixture,
) -> None:
    event = AuditEvent(
        event_type="configuration.validated",
        request_id="request-synthetic-1",
        outcome="success",
        actor_ref="synthetic.patient@example.org",
        resource_type="document",
        resource_ref="MRN:SYN-12345",
    )
    caplog.set_level(logging.INFO, logger="app.core.audit")

    await LoggingAuditSink().record(event)

    assert "synthetic.patient@example.org" not in caplog.text
    assert "SYN-12345" not in caplog.text
    assert "configuration.validated" in caplog.text

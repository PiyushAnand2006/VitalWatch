"""Audit service interface and non-sensitive logging sink."""

import logging
from typing import Protocol

from app.core.request_context import get_request_id
from app.schemas.audit import AuditEvent

logger = logging.getLogger(__name__)


class AuditSink(Protocol):
    """Destination for structured audit events."""

    async def record(self, event: AuditEvent) -> None:
        """Persist or emit one validated audit event."""
        ...


class LoggingAuditSink:
    """Stage 1 sink that emits only allow-listed audit fields."""

    async def record(self, event: AuditEvent) -> None:
        logger.info(
            "audit event_type=%s outcome=%s request_id=%s resource_type=%s reason_code=%s",
            event.event_type,
            event.outcome,
            event.request_id,
            event.resource_type or "-",
            event.reason_code or "-",
        )


class AuditService:
    """Construct validated events and delegate storage to a configured sink."""

    def __init__(self, sink: AuditSink) -> None:
        self._sink = sink

    async def record(
        self,
        event_type: str,
        outcome: str,
        *,
        request_id: str | None = None,
        actor_ref: str | None = None,
        resource_type: str | None = None,
        resource_ref: str | None = None,
        reason_code: str | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_type=event_type,
            outcome=outcome,
            request_id=request_id or get_request_id(),
            actor_ref=actor_ref,
            resource_type=resource_type,
            resource_ref=resource_ref,
            reason_code=reason_code,
        )
        await self._sink.record(event)
        return event

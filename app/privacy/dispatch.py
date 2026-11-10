"""Single-use dispatch guard for destination-bound external payloads."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar
from uuid import UUID

from app.privacy.digest import destination_sha256, payload_sha256
from app.privacy.store import PrivacyApprovalStore
from app.schemas.privacy import (
    DispatchOutcome,
    ExternalDestination,
    PreparedPayload,
    PrivacyStatus,
    utc_now,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

Transport = Callable[[PreparedPayload, ExternalDestination], Awaitable[T]]


@dataclass(frozen=True)
class DispatchResult(Generic[T]):
    """The dispatch outcome and, when allowed, the transport result."""

    outcome: DispatchOutcome
    value: T | None = None


class ApprovedDispatcher:
    """Make unapproved or altered outbound transmission impossible through this API."""

    def __init__(
        self,
        *,
        approvals: PrivacyApprovalStore,
        local_only: bool,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._approvals = approvals
        self._local_only = local_only
        self._clock = clock

    async def dispatch(
        self,
        *,
        approval_id: UUID,
        payload: PreparedPayload,
        destination: ExternalDestination,
        transport: Transport[T],
    ) -> DispatchResult[T]:
        """Verify every binding, consume the approval once, then call the transport."""

        if self._local_only:
            return DispatchResult(DispatchOutcome.BLOCKED_LOCAL_ONLY)

        digest = payload_sha256(payload)
        destination_digest = destination_sha256(destination)
        try:
            record = await self._approvals.get(approval_id)
        except Exception as exc:
            logger.error("privacy dispatch store failure type=%s", type(exc).__name__)
            return DispatchResult(DispatchOutcome.BLOCKED_STORE_FAILURE)

        if record is None or record.status is not PrivacyStatus.APPROVED:
            return DispatchResult(DispatchOutcome.BLOCKED_NOT_APPROVED)
        if record.payload_sha256 != digest:
            return DispatchResult(DispatchOutcome.BLOCKED_DIGEST_MISMATCH)
        if record.destination_digest != destination_digest:
            return DispatchResult(DispatchOutcome.BLOCKED_DESTINATION_MISMATCH)

        now = self._clock()
        if record.expires_at is None or record.expires_at <= now:
            return DispatchResult(DispatchOutcome.BLOCKED_EXPIRED)

        try:
            consumed = await self._approvals.consume(
                approval_id,
                payload_sha256=digest,
                destination_digest=destination_digest,
                now=now,
            )
        except Exception as exc:
            logger.error("privacy dispatch consume failure type=%s", type(exc).__name__)
            return DispatchResult(DispatchOutcome.BLOCKED_STORE_FAILURE)
        if not consumed:
            return DispatchResult(DispatchOutcome.BLOCKED_CONSUMED)

        value = await transport(payload, destination)
        return DispatchResult(DispatchOutcome.DISPATCHED, value)

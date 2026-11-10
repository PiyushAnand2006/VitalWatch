"""Persistence protocol for minimal privacy approval records."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID

from app.schemas.privacy import PrivacyAuditRecord


class PrivacyApprovalStore(Protocol):
    """Storage boundary used by the gate and the dispatch guard."""

    async def record(self, record: PrivacyAuditRecord) -> None:
        """Persist one privacy decision without raw payloads or PII values."""

        ...

    async def get(self, approval_id: UUID) -> PrivacyAuditRecord | None:
        """Load one approval record by identifier."""

        ...

    async def consume(
        self,
        approval_id: UUID,
        *,
        payload_sha256: str,
        destination_digest: str,
        now: datetime,
    ) -> bool:
        """Atomically mark an approval used if every binding still matches."""

        ...

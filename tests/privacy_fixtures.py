"""Synthetic evidence and in-memory stores for privacy tests."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any
from uuid import UUID

from app.privacy.engine import PiiEntity
from app.schemas.evidence import (
    EvidenceFlag,
    EvidencePackage,
    EvidenceSource,
    LabFinding,
    ObservationStatus,
    ParsedObservation,
)
from app.schemas.retrieval import RetrievalFilter, RetrievalStatus

DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")
CONTENT_HASH = "a" * 64
PATIENT_ID = "synthetic-patient-john-smith-001"


def make_source(*, quality: str = "high") -> EvidenceSource:
    return EvidenceSource(
        document_id=DOCUMENT_ID,
        chunk_id="chunk-1",
        page_number=1,
        content_hash=CONTENT_HASH,
        method="native_pdf",
        quality=quality,
        document_created_at=datetime(2026, 1, 1, tzinfo=UTC),
        score=0.91,
    )


def make_observation(
    *,
    canonical_name: str = "Potassium",
    raw_name: str = "Potassium",
    value: str | None = "5.4",
    unit: str | None = "mmol/L",
    low: str | None = "3.5",
    high: str | None = "5.1",
    observed_date: date | None = date(2026, 1, 5),
    source: EvidenceSource | None = None,
    quality: str = "high",
) -> ParsedObservation:
    resolved_source = source or make_source(quality=quality)
    identity = "|".join(
        [
            str(resolved_source.document_id),
            resolved_source.chunk_id,
            canonical_name,
            value or "",
            unit or "",
            observed_date.isoformat() if observed_date is not None else "",
        ]
    )
    evidence_id = f"ev_{sha256(identity.encode('utf-8')).hexdigest()[:24]}"
    flags = [EvidenceFlag.OUTSIDE_REFERENCE_RANGE] if value is not None else []
    return ParsedObservation(
        evidence_id=evidence_id,
        raw_name=raw_name,
        canonical_name=canonical_name,
        value_lexeme=value,
        value_decimal=Decimal(value) if value is not None else None,
        unit_lexeme=unit,
        reference_range_lexeme=(
            f"{low}-{high} {unit}" if low is not None and high is not None else None
        ),
        reference_low_lexeme=low,
        reference_high_lexeme=high,
        reference_low=Decimal(low) if low is not None else None,
        reference_high=Decimal(high) if high is not None else None,
        reference_unit_lexeme=unit if low is not None else None,
        outside_reference_range=True if value is not None and low is not None else None,
        status=(
            ObservationStatus.UNCERTAIN
            if quality == "uncertain"
            else ObservationStatus.EXPLICIT
        ),
        flags=flags if quality == "high" else [*flags, EvidenceFlag.OCR_UNCERTAIN],
        observed_date_lexeme=observed_date.isoformat() if observed_date is not None else None,
        observed_date=observed_date,
        source=resolved_source,
    )


def make_evidence(
    *,
    observations: list[ParsedObservation] | None = None,
    findings: list[LabFinding] | None = None,
    patient_id: str = PATIENT_ID,
    as_of: datetime | None = datetime(2026, 11, 10, tzinfo=UTC),
) -> EvidencePackage:
    resolved_observations = observations if observations is not None else [make_observation()]
    sources = list({observation.source for observation in resolved_observations})
    return EvidencePackage(
        query="synthetic query",
        patient_id=patient_id,
        as_of=as_of,
        retrieval_status=RetrievalStatus.RETRIEVED,
        filters=RetrievalFilter(as_of=as_of),
        observations=resolved_observations,
        findings=findings or [],
        conflicts=[],
        missing_information=[],
        excluded_after_as_of=[],
        sources=sources,
    )


def entity_for(
    text: str, value: str, entity_type: str, *, score: float = 0.9, occurrence: int = 0
) -> PiiEntity:
    start = -1
    cursor = 0
    for _ in range(occurrence + 1):
        start = text.index(value, cursor)
        cursor = start + len(value)
    return PiiEntity(
        entity_type=entity_type,
        start=start,
        end=start + len(value),
        score=score,
    )


class MemoryApprovalStore:
    """Deterministic store mirroring the repository's consume semantics."""

    def __init__(self) -> None:
        self.records: dict[UUID, Any] = {}
        self.consumed: set[UUID] = set()
        self.record_error: Exception | None = None
        self.get_error: Exception | None = None
        self.consume_error: Exception | None = None

    async def record(self, record: Any) -> None:
        if self.record_error is not None:
            raise self.record_error
        self.records[record.approval_id] = record

    async def get(self, approval_id: UUID) -> Any | None:
        if self.get_error is not None:
            raise self.get_error
        return self.records.get(approval_id)

    async def consume(
        self,
        approval_id: UUID,
        *,
        payload_sha256: str,
        destination_digest: str,
        now: datetime,
    ) -> bool:
        if self.consume_error is not None:
            raise self.consume_error
        record = self.records.get(approval_id)
        if (
            record is None
            or record.payload_sha256 != payload_sha256
            or record.destination_digest != destination_digest
            or record.expires_at is None
            or record.expires_at <= now
            or approval_id in self.consumed
            or record.status.value != "approved"
        ):
            return False
        self.consumed.add(approval_id)
        return True


class StaticAnalyzer:
    """Analyzer that returns fixed entities; optionally fails."""

    def __init__(
        self,
        entities: dict[str, list[PiiEntity]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._entities = entities or {}
        self._error = error
        self.calls = 0

    def analyze(self, text: str) -> list[PiiEntity]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return list(self._entities.get(text, []))

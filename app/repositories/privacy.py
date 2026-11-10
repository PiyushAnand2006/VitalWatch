"""PostgreSQL repository for minimal, PII-free privacy approval records."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import PrivacyApprovalModel
from app.schemas.privacy import (
    DispatchPurpose,
    PrivacyAuditRecord,
    PrivacyStatus,
    QuestionType,
)


def _to_record(row: PrivacyApprovalModel) -> PrivacyAuditRecord:
    return PrivacyAuditRecord(
        approval_id=row.id,
        request_id=row.request_id,
        status=PrivacyStatus(row.status),
        purpose=DispatchPurpose(row.purpose),
        provider=row.provider,
        model=row.model_name,
        destination_digest=row.destination_digest,
        payload_sha256=row.payload_sha256,
        patient_ref=row.patient_ref,
        actor_ref=row.actor_ref,
        question_type=QuestionType(row.question_type),
        observation_count=row.observation_count,
        reason_codes=list(row.reason_codes or []),
        review_flags=list(row.review_flags or []),
        entity_counts=dict(row.entity_counts or {}),
        created_at=row.created_at,
        expires_at=row.expires_at,
    )


class PrivacyApprovalRepository:
    """Store approval metadata and atomically consume approved records."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def record(self, record: PrivacyAuditRecord) -> None:
        row = PrivacyApprovalModel(
            id=record.approval_id,
            request_id=record.request_id,
            status=record.status.value,
            purpose=record.purpose.value,
            provider=record.provider,
            model_name=record.model,
            destination_digest=record.destination_digest,
            payload_sha256=record.payload_sha256,
            patient_ref=record.patient_ref,
            actor_ref=record.actor_ref,
            question_type=record.question_type.value,
            observation_count=record.observation_count,
            reason_codes=list(record.reason_codes),
            review_flags=list(record.review_flags),
            entity_counts=dict(record.entity_counts),
            created_at=record.created_at,
            expires_at=record.expires_at,
        )
        async with self._sessions() as session:
            session.add(row)
            await session.commit()

    async def get(self, approval_id: UUID) -> PrivacyAuditRecord | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(PrivacyApprovalModel).where(PrivacyApprovalModel.id == approval_id)
            )
        return _to_record(row) if row is not None else None

    async def consume(
        self,
        approval_id: UUID,
        *,
        payload_sha256: str,
        destination_digest: str,
        now: datetime,
    ) -> bool:
        statement = (
            update(PrivacyApprovalModel)
            .where(
                PrivacyApprovalModel.id == approval_id,
                PrivacyApprovalModel.status == PrivacyStatus.APPROVED.value,
                PrivacyApprovalModel.payload_sha256 == payload_sha256,
                PrivacyApprovalModel.destination_digest == destination_digest,
                PrivacyApprovalModel.consumed_at.is_(None),
                PrivacyApprovalModel.expires_at.is_not(None),
                PrivacyApprovalModel.expires_at > now,
            )
            .values(consumed_at=now)
            .returning(PrivacyApprovalModel.id)
        )
        async with self._sessions() as session:
            consumed = (await session.execute(statement)).scalar_one_or_none() is not None
            await session.commit()
        return consumed

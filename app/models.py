"""PostgreSQL persistence models for patient-scoped Stage 2 resources."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class PatientGrantModel(Base):
    """A server-controlled grant; a token never conveys patient access itself."""

    __tablename__ = "patient_grants"
    __table_args__ = (UniqueConstraint("user_id", "patient_id", name="uq_patient_grant"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    patient_id: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class DocumentModel(Base):
    """Protected metadata for an uploaded original kept in private storage."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("patient_id", "checksum", name="uq_document_patient_checksum"),
        Index("ix_documents_patient_created", "patient_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    patient_id: Mapped[str] = mapped_column(String(128), index=True)
    uploader_id: Mapped[str] = mapped_column(String(128))
    source_name: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(128), unique=True)
    media_type: Mapped[str] = mapped_column(String(128))
    checksum: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int]
    status: Mapped[str] = mapped_column(String(16), default="uploaded")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

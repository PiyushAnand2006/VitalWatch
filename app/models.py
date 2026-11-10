"""PostgreSQL persistence models for patient-scoped resources and chunk vectors."""

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.schemas.embeddings import EMBEDDING_DIMENSION


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


class PrivacyApprovalModel(Base):
    """Minimal privacy decision metadata with no raw payload or PII values."""

    __tablename__ = "privacy_approvals"
    __table_args__ = (
        Index("ix_privacy_approvals_status_created", "status", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    request_id: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    purpose: Mapped[str] = mapped_column(String(32))
    provider: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column("model", String(128))
    destination_digest: Mapped[str] = mapped_column(String(64))
    payload_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    patient_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    actor_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    question_type: Mapped[str] = mapped_column(String(32))
    observation_count: Mapped[int]
    reason_codes: Mapped[list[str]] = mapped_column(JSON, default=list)
    review_flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    entity_counts: Mapped[dict[str, int]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    consumed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class ChunkEmbeddingModel(Base):
    """A private chunk vector bound to one patient, document, and model identity."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_id", name="uq_chunk_embedding_document_chunk"),
        Index("ix_chunk_embeddings_patient_document", "patient_id", "document_id"),
        Index("ix_chunk_embeddings_model", "model_name", "model_revision", "dimension"),
        Index(
            "ix_chunk_embeddings_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    patient_id: Mapped[str] = mapped_column(String(128), index=True)
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_id: Mapped[str] = mapped_column(String(64))
    content_hash: Mapped[str] = mapped_column(String(64))
    page_number: Mapped[int]
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    method: Mapped[str] = mapped_column(String(32))
    quality: Mapped[str] = mapped_column(String(16))
    model_name: Mapped[str] = mapped_column(String(255))
    model_revision: Mapped[str] = mapped_column(String(128))
    dimension: Mapped[int]
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIMENSION))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

"""Strict contracts for patient-scoped retrieval and typed no-evidence results."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.embeddings import EmbeddingModelInfo


class RetrievalStatus(StrEnum):
    RETRIEVED = "retrieved"
    NO_EVIDENCE = "no_evidence"


class NoEvidenceReason(StrEnum):
    NO_MATCHING_DOCUMENTS = "no_matching_documents"
    BELOW_RELEVANCE_THRESHOLD = "below_relevance_threshold"


class RetrievalFilter(BaseModel):
    """Server-built filters; no client-supplied SQL or patient widening."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of: datetime | None = None
    document_ids: tuple[UUID, ...] | None = None
    created_from: datetime | None = None
    created_to: datetime | None = None

    @field_validator("as_of", "created_from", "created_to")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("retrieval timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def reject_inverted_window(self) -> "RetrievalFilter":
        if (
            self.created_from is not None
            and self.created_to is not None
            and self.created_from > self.created_to
        ):
            raise ValueError("created_from must not be after created_to")
        return self


class RetrievedChunk(BaseModel):
    """One untrusted source chunk with resolved provenance and relevance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1, max_length=64)
    document_id: UUID
    patient_id: str = Field(min_length=1, max_length=128)
    page_number: int = Field(ge=1)
    section: str | None = None
    text: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    method: str
    quality: str = Field(pattern=r"^(high|uncertain)$")
    score: float
    distance: float = Field(ge=0)
    document_created_at: datetime
    trust_level: Literal["untrusted_retrieved_text"] = "untrusted_retrieved_text"


class RetrievalResult(BaseModel):
    """Deterministic retrieval outcome, including the explicit no-evidence state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    patient_id: str = Field(min_length=1, max_length=128)
    status: RetrievalStatus
    reason: NoEvidenceReason | None = None
    items: list[RetrievedChunk]
    candidates_considered: int = Field(ge=0)
    duplicates_removed: int = Field(ge=0)
    below_threshold: int = Field(ge=0)
    top_k: int = Field(ge=1)
    min_score: float = Field(ge=0.0, le=1.0)
    model: EmbeddingModelInfo
    filters: RetrievalFilter

    @model_validator(mode="after")
    def enforce_status_consistency(self) -> "RetrievalResult":
        if self.status is RetrievalStatus.RETRIEVED:
            if not self.items or self.reason is not None:
                raise ValueError("retrieved results require items and no no-evidence reason")
        elif self.items or self.reason is None:
            raise ValueError("no-evidence results require a reason and no items")
        if any(item.patient_id != self.patient_id for item in self.items):
            raise ValueError("retrieved items must belong to the requested patient")
        return self

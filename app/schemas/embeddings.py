"""Private embedding, indexing, and vector-search contracts."""

import math
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

EMBEDDING_DIMENSION = 384
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModelInfo(BaseModel):
    """Identity that must match for every vector stored in the same index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=255)
    revision: str = Field(min_length=1, max_length=128)
    dimension: int = Field(ge=1, le=2000)


class EmbeddedChunk(BaseModel):
    """A chunk plus its locally computed vector; retained only inside PostgreSQL."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1, max_length=64)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    page_number: int = Field(ge=1)
    section: str | None = Field(default=None, max_length=255)
    text: str = Field(min_length=1)
    method: str = Field(min_length=1, max_length=32)
    quality: str = Field(pattern=r"^(high|uncertain)$")
    embedding: list[float] = Field(min_length=EMBEDDING_DIMENSION, max_length=EMBEDDING_DIMENSION)

    @field_validator("embedding")
    @classmethod
    def reject_non_finite_values(cls, value: list[float]) -> list[float]:
        """Reject NaN and infinite values before they reach pgvector."""

        if not all(math.isfinite(component) for component in value):
            raise ValueError("embedding values must be finite")
        return value


class ChunkMatch(BaseModel):
    """A patient-scoped vector search result with source provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    document_id: UUID
    patient_id: str
    page_number: int = Field(ge=1)
    section: str | None = None
    text: str
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    method: str
    quality: str
    distance: float = Field(ge=0)
    score: float
    document_created_at: datetime


class IndexRebuildReport(BaseModel):
    """Outcome of a controlled, cross-document index rebuild."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: EmbeddingModelInfo
    cleared_vectors: int = Field(ge=0)
    indexed_documents: int = Field(ge=0)
    chunks_indexed: int = Field(ge=0)
    failed_document_ids: list[UUID]


class IndexResult(BaseModel):
    """Outcome of a server-controlled document index operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: UUID
    chunks_indexed: int = Field(ge=0)
    model: EmbeddingModelInfo

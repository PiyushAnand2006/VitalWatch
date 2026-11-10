"""Public document contracts. Storage keys and host paths are intentionally absent."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    QUARANTINED = "quarantined"


class DocumentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: UUID
    patient_id: str = Field(min_length=1, max_length=128)
    source_name: str = Field(min_length=1, max_length=255)
    media_type: str = Field(min_length=1, max_length=128)
    size_bytes: int = Field(ge=0)
    checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime


class DocumentPage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: list[DocumentSummary]
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    total: int = Field(ge=0)

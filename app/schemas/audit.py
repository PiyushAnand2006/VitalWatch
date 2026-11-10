"""Minimal audit event contract that excludes free-form sensitive payloads."""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AuditEvent(BaseModel):
    """Structured audit metadata; raw content and arbitrary metadata are not accepted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: UUID = Field(default_factory=uuid4)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,63}$")
    request_id: str = Field(min_length=1, max_length=128)
    outcome: Literal["success", "failure", "blocked"]
    actor_ref: str | None = Field(default=None, max_length=128)
    resource_type: str | None = Field(default=None, max_length=64)
    resource_ref: str | None = Field(default=None, max_length=128)
    reason_code: str | None = Field(default=None, max_length=64)

    @field_validator("occurred_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        """Require an explicit timezone for portable audit timestamps."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("audit timestamps must be timezone-aware")
        return value

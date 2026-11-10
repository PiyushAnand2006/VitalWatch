"""Health and readiness response contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Liveness response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    service: str
    version: str
    environment: str
    request_id: str


class ReadinessCheck(BaseModel):
    """One non-sensitive dependency readiness result."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: Literal["passed", "failed"]
    message: str


class ReadyResponse(BaseModel):
    """Aggregate readiness response."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "not_ready"]
    service: str
    version: str
    checks: list[ReadinessCheck]
    request_id: str

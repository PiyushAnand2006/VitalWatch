"""Safe public error response contracts."""

from pydantic import BaseModel, ConfigDict, Field


class ValidationIssue(BaseModel):
    """A validation issue with input values and internal context removed."""

    model_config = ConfigDict(extra="forbid")

    location: list[str]
    message: str
    type: str


class ErrorBody(BaseModel):
    """Public error information."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    issues: list[ValidationIssue] | None = None


class ErrorResponse(BaseModel):
    """Consistent error envelope returned to clients."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorBody
    request_id: str = Field(min_length=1, max_length=128)

"""Schema verifier: strict Pydantic parsing of external model output."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import ValidationError

from app.schemas.orchestration import AdjudicationCandidate, CritiqueOutput, ReasoningCandidate

_FENCE = "```"


class SchemaError(ValueError):
    """Model output is not schema-valid JSON; one repair attempt is allowed."""


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model text, tolerating code fences."""

    stripped = text.strip()
    if stripped.startswith(_FENCE):
        first_newline = stripped.find("\n")
        closing = stripped.rfind(_FENCE)
        if first_newline != -1 and closing > first_newline:
            stripped = stripped[first_newline + 1 : closing].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise SchemaError("no JSON object found") from None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SchemaError(f"invalid JSON: {exc.msg}") from None
    if not isinstance(parsed, dict):
        raise SchemaError("JSON payload is not an object")
    return parsed


ModelT = TypeVar("ModelT", ReasoningCandidate, CritiqueOutput, AdjudicationCandidate)


def _parse(text: str, model: type[ModelT]) -> ModelT:
    """Parse model output against a strict schema; raises SchemaError on failure."""

    try:
        return model.model_validate(extract_json_object(text))
    except ValidationError as exc:
        raise SchemaError(_first_error(exc)) from None


def parse_candidate(text: str) -> ReasoningCandidate:
    """Parse one reasoner output; raises SchemaError on malformed JSON or schema."""

    return _parse(text, ReasoningCandidate)


def parse_critique(text: str) -> CritiqueOutput:
    """Parse one reviewer critique; raises SchemaError on malformed output."""

    return _parse(text, CritiqueOutput)


def parse_adjudication(text: str) -> AdjudicationCandidate:
    """Parse the adjudicator synthesis; raises SchemaError on malformed output."""

    return _parse(text, AdjudicationCandidate)


def _first_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error.get("loc", ()))
    return f"{location or 'payload'}: {error.get('msg', 'invalid')}"

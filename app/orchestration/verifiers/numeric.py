"""Numeric/unit verifier: exact values, decimals, signs, and units against the payload."""

from __future__ import annotations

import re

from app.orchestration.verifiers.base import extract_measured_numbers
from app.schemas.orchestration import VerifierName
from app.schemas.orchestration import VerifierResult as Result
from app.schemas.privacy import MinimizedObservation

_RANGE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:-|–|—|to|through)\s*([+-]?\d+(?:\.\d+)?)\s*(.*)$",
    re.IGNORECASE,
)


def _allowed_values(observation: MinimizedObservation) -> list[tuple[str, str | None]]:
    """Value and reference-bound lexemes the claim may legitimately state."""

    allowed: list[tuple[str, str | None]] = []
    if observation.value is not None:
        allowed.append((observation.value, observation.unit))
    if observation.reference_range is not None:
        match = _RANGE.match(observation.reference_range)
        if match is not None:
            low, high, unit = match.group(1), match.group(2), match.group(3).strip()
            allowed.append((low, unit or None))
            allowed.append((high, unit or None))
    return allowed


def verify_numeric(
    *,
    claim_text: str,
    cited_observations: list[MinimizedObservation],
) -> Result:
    """Every decimal number and every number-with-unit must match the payload exactly."""

    measured = extract_measured_numbers(claim_text)
    if not measured:
        return Result(verifier=VerifierName.NUMERIC, passed=True)

    allowed: list[tuple[str, str | None]] = []
    for observation in cited_observations:
        allowed.extend(_allowed_values(observation))
    allowed_units = {unit for _, unit in allowed if unit is not None}

    matched: list[str] = []
    for number in measured:
        # A unit candidate counts as a real unit only when it carries a unit-ish
        # character (/, %, µ) or matches a payload unit; English words that merely
        # follow a number are downgraded so the value is checked on its own.
        has_real_unit = number.unit is not None and (
            not number.unit.isalpha() or number.unit in allowed_units
        )
        effective_unit = number.unit if has_real_unit else None
        if effective_unit is None and not number.has_decimal:
            # Bare integers (counts, ordinal wording) are not verifiable claims.
            continue
        hit = next(
            (
                (value, unit)
                for value, unit in allowed
                if value == number.lexeme
                and (effective_unit is None or unit == effective_unit)
            ),
            None,
        )
        if hit is None:
            return Result(
                verifier=VerifierName.NUMERIC,
                passed=False,
                reason_codes=["value_mismatch"],
            )
        matched.append(
            number.lexeme
            if effective_unit is None
            else f"{number.lexeme} {effective_unit}"
        )
    return Result(
        verifier=VerifierName.NUMERIC, passed=True, matched_values=matched
    )

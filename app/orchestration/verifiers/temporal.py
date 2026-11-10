"""Temporal verifier: dates in claims must come from the cited evidence."""

from __future__ import annotations

from app.orchestration.verifiers.base import extract_year_months
from app.schemas.orchestration import VerifierName
from app.schemas.orchestration import VerifierResult as Result
from app.schemas.privacy import MinimizedObservation


def verify_temporal(
    *,
    claim_text: str,
    cited_observations: list[MinimizedObservation],
    as_of_year_month: tuple[int, int] | None,
) -> Result:
    """Reject invented dates and any date after the as-of boundary."""

    mentioned = extract_year_months(claim_text)
    if not mentioned:
        return Result(verifier=VerifierName.TEMPORAL, passed=True)

    allowed = {
        (year, month)
        for observation in cited_observations
        if observation.date is not None
        for year, month in [_parse_year_month(observation.date)]
        if (year, month) != (-1, -1)
    }
    for year, month in mentioned:
        if (year, month) not in allowed:
            return Result(
                verifier=VerifierName.TEMPORAL,
                passed=False,
                reason_codes=["unsupported_date"],
            )
        if as_of_year_month is not None and (year, month) > as_of_year_month:
            return Result(
                verifier=VerifierName.TEMPORAL,
                passed=False,
                reason_codes=["date_after_as_of"],
            )
    return Result(verifier=VerifierName.TEMPORAL, passed=True)


def _parse_year_month(value: str) -> tuple[int, int]:
    parts = value.split("-")
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return (-1, -1)
    year, month = int(parts[0]), int(parts[1])
    if not 1 <= month <= 12:
        return (-1, -1)
    return (year, month)

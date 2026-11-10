"""Consistency verifier: internal contradictions and conflict preservation."""

from __future__ import annotations

from app.orchestration.verifiers.base import extract_measured_numbers
from app.schemas.orchestration import CandidateClaim, ReasoningCandidate, VerifierName
from app.schemas.orchestration import VerifierResult as Result
from app.schemas.privacy import MinimizedConflict, MinimizedObservation


def verify_consistency(
    *,
    claim: CandidateClaim,
    candidate: ReasoningCandidate,
    cited_observations: list[MinimizedObservation],
    payload_conflicts: list[MinimizedConflict],
) -> Result:
    """A value asserted for a source-conflicted lab must be surfaced as a conflict.

    Silently resolving a genuine source conflict is prohibited: a claim may state
    one source value only when the candidate also reports the conflict exists.
    """

    conflicted_values = {conflict.lab: set(conflict.values) for conflict in payload_conflicts}
    surfaced_labs = {conflict.lab for conflict in candidate.conflicts}
    for observation in cited_observations:
        values = conflicted_values.get(observation.lab)
        if not values or observation.value is None:
            continue
        asserted = {
            number.lexeme
            for number in extract_measured_numbers(claim.text)
            if number.has_decimal and number.lexeme == observation.value
        }
        if asserted and observation.lab not in surfaced_labs:
            return Result(
                verifier=VerifierName.CONSISTENCY,
                passed=False,
                reason_codes=["conflict_not_surfaced"],
            )
    return Result(verifier=VerifierName.CONSISTENCY, passed=True)


def find_internal_contradictions(
    candidate: ReasoningCandidate,
    ref_labs: dict[str, frozenset[str]],
    ref_values: dict[str, frozenset[tuple[str, str | None]]],
) -> list[str]:
    """Detect one lab asserted with two different exact source values.

    Only (value, unit) pairs that exactly match a cited observation's own value
    count as assertions; reference-range bounds and other payload numbers are not
    competing value claims.
    """

    asserted: dict[str, set[str]] = {}
    for claim in candidate.claims:
        labs = {
            lab
            for ref in claim.evidence_refs
            for lab in ref_labs.get(ref, frozenset())
        }
        allowed = {
            (lexeme, unit)
            for ref in claim.evidence_refs
            for lexeme, unit in ref_values.get(ref, frozenset())
        }
        if not labs or not allowed:
            continue
        for number in extract_measured_numbers(claim.text):
            if number.unit is None:
                continue
            pair = (number.lexeme, number.unit)
            if pair not in allowed:
                continue
            for lab in labs:
                asserted.setdefault(lab, set()).add(f"{number.lexeme} {number.unit}")
    return [
        f"{lab}: {', '.join(sorted(values))}"
        for lab, values in asserted.items()
        if len(values) > 1
    ]

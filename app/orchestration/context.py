"""Local mapping between approved payload tokens and authorized evidence provenance.

The privacy gate tokenizes evidence IDs ([EV_1], [EV_2], ...). This module rebuilds
the token-to-evidence mapping deterministically from the payload and evidence
package alone, failing closed: a reference that cannot be matched to an authorized
observation resolves to nothing and any claim citing it is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.orchestration.verifiers.consistency import (
    find_internal_contradictions,
    verify_consistency,
)
from app.orchestration.verifiers.evidence import verify_evidence
from app.orchestration.verifiers.numeric import verify_numeric
from app.orchestration.verifiers.provenance import verify_provenance
from app.orchestration.verifiers.safety import verify_safety
from app.orchestration.verifiers.temporal import verify_temporal
from app.privacy.temporal import generalize_observed_date
from app.schemas.evidence import EvidencePackage, ParsedObservation
from app.schemas.orchestration import (
    CandidateClaim,
    ClaimVerdict,
    ReasoningCandidate,
    VerifierName,
    VerifierResult,
)
from app.schemas.privacy import MinimizedObservation, PreparedPayload


@dataclass(frozen=True)
class VerificationContext:
    """Everything the deterministic verifiers need, resolved locally."""

    payload: PreparedPayload
    evidence: EvidencePackage
    known_refs: frozenset[str]
    ref_labs: dict[str, frozenset[str]]
    ref_observations: dict[str, tuple[MinimizedObservation, ...]]
    token_map: dict[str, frozenset[str]]
    known_source_keys: frozenset[tuple[str, str, int]]
    ref_values: dict[str, frozenset[tuple[str, str | None]]]
    as_of_year_month: tuple[int, int] | None


def build_verification_context(
    payload: PreparedPayload, evidence: EvidencePackage
) -> VerificationContext:
    """Match payload references to evidence observations; unmatched refs stay empty."""

    token_map: dict[str, frozenset[str]] = {}
    ref_observations: dict[str, tuple[MinimizedObservation, ...]] = {}
    ref_labs: dict[str, set[str]] = {}
    for observation in payload.observations:
        matched = frozenset(
            candidate.evidence_id
            for candidate in evidence.observations
            if _matches(observation, candidate)
        )
        token_map[observation.ref] = matched
        ref_observations[observation.ref] = (observation,)
        ref_labs.setdefault(observation.ref, set()).add(observation.lab)
    known_refs = set(token_map)
    for finding in payload.findings:
        known_refs.update(finding.refs)
        for ref in finding.refs:
            ref_labs.setdefault(ref, set()).add(finding.lab)
    for conflict in payload.conflicts:
        known_refs.update(conflict.refs)
        for ref in conflict.refs:
            ref_labs.setdefault(ref, set()).add(conflict.lab)

    known_source_keys = frozenset(
        (str(source.document_id), source.chunk_id, source.page_number)
        for source in evidence.sources
    )
    as_of = evidence.as_of
    as_of_year_month = (as_of.year, as_of.month) if as_of is not None else None
    return VerificationContext(
        payload=payload,
        evidence=evidence,
        known_refs=frozenset(known_refs),
        ref_labs={ref: frozenset(labs) for ref, labs in ref_labs.items()},
        ref_observations=ref_observations,
        token_map=token_map,
        known_source_keys=known_source_keys,
        ref_values={
            ref: frozenset(
                (observation.value, observation.unit)
                for observation in observations
                if observation.value is not None
            )
            for ref, observations in ref_observations.items()
        },
        as_of_year_month=as_of_year_month,
    )


def verify_claim(
    *,
    claim: CandidateClaim,
    candidate: ReasoningCandidate,
    context: VerificationContext,
    is_fact: bool,
) -> ClaimVerdict:
    """Run every deterministic verifier against one claim."""

    cited = [
        observation
        for ref in claim.evidence_refs
        for observation in context.ref_observations.get(ref, ())
    ]
    results: list[VerifierResult] = [
        VerifierResult(verifier=VerifierName.SCHEMA, passed=True),
        verify_evidence(
            claim_text=claim.text,
            evidence_refs=claim.evidence_refs,
            known_refs=context.known_refs,
            ref_labs=context.ref_labs,
        ),
        verify_numeric(claim_text=claim.text, cited_observations=cited),
        verify_temporal(
            claim_text=claim.text,
            cited_observations=cited,
            as_of_year_month=context.as_of_year_month,
        ),
        verify_consistency(
            claim=claim,
            candidate=candidate,
            cited_observations=cited,
            payload_conflicts=context.payload.conflicts,
        ),
        verify_safety(claim_text=claim.text),
        verify_provenance(
            evidence_refs=claim.evidence_refs,
            token_map=context.token_map,
            evidence=context.evidence,
            known_source_keys=context.known_source_keys,
        ),
    ]
    return ClaimVerdict(
        claim_text=claim.text,
        evidence_refs=claim.evidence_refs,
        results=tuple(results),
        is_fact=is_fact,
    )


def candidate_contradictions(
    candidate: ReasoningCandidate, context: VerificationContext
) -> list[str]:
    """Internal value contradictions for one candidate, keyed by lab."""

    return find_internal_contradictions(candidate, context.ref_labs, context.ref_values)


def _matches(payload_observation: MinimizedObservation, candidate: ParsedObservation) -> bool:
    """Match on the fields the payload preserved; absent fields are not compared."""

    if payload_observation.lab != candidate.canonical_name:
        return False
    if payload_observation.status != candidate.status.value:
        return False
    if (
        payload_observation.value is not None
        and payload_observation.value != candidate.value_lexeme
    ):
        return False
    if (
        payload_observation.unit is not None
        and payload_observation.unit != candidate.unit_lexeme
    ):
        return False
    if (
        payload_observation.date is not None
        and payload_observation.date
        != generalize_observed_date(candidate.observed_date)
    ):
        return False
    return True


__all__ = [
    "VerificationContext",
    "build_verification_context",
    "candidate_contradictions",
    "verify_claim",
]

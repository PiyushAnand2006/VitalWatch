"""Evidence verifier: citations exist in the payload and support the claim."""

from __future__ import annotations

from app.schemas.orchestration import VerifierName, VerifierResult


def verify_evidence(
    *,
    claim_text: str,
    evidence_refs: tuple[str, ...],
    known_refs: frozenset[str],
    ref_labs: dict[str, frozenset[str]],
) -> VerifierResult:
    """Reject unknown references and claims their references do not support."""

    if not evidence_refs:
        return VerifierResult(
            verifier=VerifierName.EVIDENCE, passed=False, reason_codes=["missing_citation"]
        )
    unknown = [ref for ref in evidence_refs if ref not in known_refs]
    if unknown:
        return VerifierResult(
            verifier=VerifierName.EVIDENCE,
            passed=False,
            reason_codes=["unknown_reference"],
        )
    lowered = claim_text.lower()
    supported = any(
        lab.lower() in lowered
        for ref in evidence_refs
        for lab in ref_labs.get(ref, frozenset())
    )
    if not supported:
        return VerifierResult(
            verifier=VerifierName.EVIDENCE,
            passed=False,
            reason_codes=["unsupported_reference"],
        )
    return VerifierResult(verifier=VerifierName.EVIDENCE, passed=True)

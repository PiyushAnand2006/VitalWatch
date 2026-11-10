"""Provenance verifier: every cited token resolves to authorized source provenance."""

from __future__ import annotations

from app.schemas.evidence import EvidencePackage
from app.schemas.orchestration import VerifierName
from app.schemas.orchestration import VerifierResult as Result


def verify_provenance(
    *,
    evidence_refs: tuple[str, ...],
    token_map: dict[str, frozenset[str]],
    evidence: EvidencePackage,
    known_source_keys: frozenset[tuple[str, str, int]],
) -> Result:
    """Each reference resolves to an authorized document, page, and chunk."""

    reasons: list[str] = []
    for ref in evidence_refs:
        evidence_ids = token_map.get(ref, frozenset())
        if not evidence_ids:
            reasons.append("unresolved_reference")
            continue
        for evidence_id in evidence_ids:
            source = evidence.resolve(evidence_id)
            if source is None:
                reasons.append("unresolved_provenance")
                continue
            source_key = (str(source.document_id), source.chunk_id, source.page_number)
            if source_key not in known_source_keys:
                reasons.append("unauthorized_source")
    if reasons:
        return Result(
            verifier=VerifierName.PROVENANCE,
            passed=False,
            reason_codes=list(dict.fromkeys(reasons)),
        )
    return Result(verifier=VerifierName.PROVENANCE, passed=True)

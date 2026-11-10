"""Safety verifier: diagnostic certainty and treatment instructions are rejected."""

from __future__ import annotations

import re

from app.schemas.orchestration import VerifierName
from app.schemas.orchestration import VerifierResult as Result

_CERTAINTY = re.compile(
    r"\b(?:definitely|certainly|absolutely|undoubtedly|surely|guarantee[sd]?)\b"
    r"[^.!?]{0,120}?\b(?:diagnos\w*|disease|disorder|condition|infection|cancer|"
    r"failure|syndrome|deficiency|has|have|is|are)\b",
    re.IGNORECASE,
)
_DIAGNOSIS_CERTAINTY = re.compile(
    r"\b(?:confirm\w*|rules? out)\b[^.!?]{0,80}?\b(?:diagnos\w*|disease|condition|"
    r"infection|deficiency)\b",
    re.IGNORECASE,
)
_TREATMENT = re.compile(
    r"\b(?:take|start|stop|begin|continue|increase|decrease|reduce|adjust|switch|"
    r"double|halve|resume|discontinu\w*|administer)\b\s+(?:your\s+|his\s+|her\s+|"
    r"their\s+|the\s+|a\s+|an\s+)?(?:dose|dosage|medication|medicine|meds|drug|"
    r"insulin|aspirin|therapy|treatment|supplement|antibiotic|steroid|warfarin|"
    r"metformin|statin|diuretic)",
    re.IGNORECASE,
)
_PRESCRIBE = re.compile(r"\bprescrib\w*\b", re.IGNORECASE)
_SHOULD_TAKE = re.compile(
    r"\b(?:should|must|needs? to)\s+(?:take|start|stop|begin|increase|decrease|"
    r"adjust|avoid|get)\b",
    re.IGNORECASE,
)


def verify_safety(*, claim_text: str) -> Result:
    """Reject diagnostic certainty and any medication/treatment instruction."""

    checks: list[tuple[re.Pattern[str], str]] = [
        (_CERTAINTY, "diagnostic_certainty"),
        (_DIAGNOSIS_CERTAINTY, "diagnostic_certainty"),
        (_TREATMENT, "treatment_advice"),
        (_PRESCRIBE, "treatment_advice"),
        (_SHOULD_TAKE, "treatment_advice"),
    ]
    reasons: list[str] = []
    for pattern, reason in checks:
        if pattern.search(claim_text):
            reasons.append(reason)
    if reasons:
        return Result(
            verifier=VerifierName.SAFETY,
            passed=False,
            reason_codes=list(dict.fromkeys(reasons)),
        )
    return Result(verifier=VerifierName.SAFETY, passed=True)

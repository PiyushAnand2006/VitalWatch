"""Question-specific field allow-listing, relevance, and review rules."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.schemas.evidence import (
    EvidencePackage,
    LabConflict,
    LabFinding,
    ObservationStatus,
    ParsedObservation,
)
from app.schemas.privacy import ObservationField, QuestionType
from app.services.evidence import LAB_ALIASES

_ALL_FIELDS: frozenset[ObservationField] = frozenset(ObservationField)

_FIELDS_BY_QUESTION: dict[QuestionType, frozenset[ObservationField]] = {
    # A direct value question needs the full result context, including the date.
    QuestionType.LAB_VALUE: _ALL_FIELDS,
    # Trends need values and generalized dates, not reference ranges.
    QuestionType.TREND: frozenset(
        {
            ObservationField.LAB,
            ObservationField.VALUE,
            ObservationField.UNIT,
            ObservationField.DATE,
            ObservationField.STATUS,
            ObservationField.FLAGS,
            ObservationField.EVIDENCE_REF,
        }
    ),
    # Medication questions need renal/hepatic context but not exact dates.
    QuestionType.MEDICATION: frozenset(
        {
            ObservationField.LAB,
            ObservationField.VALUE,
            ObservationField.UNIT,
            ObservationField.REFERENCE_RANGE,
            ObservationField.STATUS,
            ObservationField.FLAGS,
            ObservationField.EVIDENCE_REF,
        }
    ),
    QuestionType.DIAGNOSIS: frozenset(
        {
            ObservationField.LAB,
            ObservationField.VALUE,
            ObservationField.UNIT,
            ObservationField.REFERENCE_RANGE,
            ObservationField.STATUS,
            ObservationField.FLAGS,
            ObservationField.EVIDENCE_REF,
        }
    ),
    # General questions get the smallest useful clinical slice.
    QuestionType.GENERAL: frozenset(
        {
            ObservationField.LAB,
            ObservationField.VALUE,
            ObservationField.UNIT,
            ObservationField.STATUS,
            ObservationField.FLAGS,
            ObservationField.EVIDENCE_REF,
        }
    ),
}

_MEDICATION_TERMS = (
    "medication",
    "medicine",
    "drug",
    "dose",
    "dosage",
    "prescribed",
    "prescription",
    "contraindication",
    "interaction",
    "side effect",
    "taking",
    "mg",
)
_TREND_TERMS = (
    "trend",
    "over time",
    "changed",
    "change",
    "increase",
    "decrease",
    "rising",
    "falling",
    "previous",
    "compare",
    "serial",
)
_DIAGNOSIS_TERMS = ("diagnos", "condition", "disease", "cause", "why", "etiology")
_LAB_TERMS = ("level", "value", "result", "normal", "high", "low", "range", "mean", "interpret")

_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions?\b"),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\b"),
    re.compile(r"\b(?:system|developer)\s+prompt\b"),
    re.compile(r"\byou\s+are\s+now\s+(?:an?\s+)?(?:unrestricted|admin|dan)\b"),
    re.compile(r"\breveal\s+(?:your\s+)?(?:system|hidden|secret|instructions?)\b"),
    re.compile(r"\b(?:print|show|expose)\s+(?:the\s+)?(?:api\s+key|secret|token|password)\b"),
    re.compile(r"\boverride\s+(?:the\s+)?(?:safety|privacy|system)\b"),
)

_STRONG_IDENTIFIERS = frozenset(
    {
        "ADDRESS",
        "DATE_OF_BIRTH",
        "FACILITY",
        "GOVERNMENT_ID",
        "INTERNAL_ID",
        "MEDICAL_ID",
        "MEDICAL_RECORD_NUMBER",
        "PERSON",
    }
)


@dataclass(frozen=True)
class QuestionPolicy:
    """The deterministic minimization profile for one question."""

    question_type: QuestionType
    allowed_fields: frozenset[ObservationField]
    requested_labs: frozenset[str]


def classify_question(
    question: str, aliases: Mapping[str, str] = LAB_ALIASES
) -> QuestionPolicy:
    """Classify a question and derive its allow-listed lab scope."""

    lowered = question.casefold()
    requested = _requested_labs(lowered, aliases)
    if _has_term(lowered, _MEDICATION_TERMS):
        question_type = QuestionType.MEDICATION
    elif _has_term(lowered, _TREND_TERMS):
        question_type = QuestionType.TREND
    elif _has_term(lowered, _DIAGNOSIS_TERMS):
        question_type = QuestionType.DIAGNOSIS
    elif requested or _has_term(lowered, _LAB_TERMS):
        question_type = QuestionType.LAB_VALUE
    else:
        question_type = QuestionType.GENERAL
    return QuestionPolicy(
        question_type=question_type,
        allowed_fields=_FIELDS_BY_QUESTION[question_type],
        requested_labs=requested,
    )


def detect_prompt_injection(question: str) -> bool:
    """Flag instruction-like text in untrusted questions for human review."""

    lowered = question.casefold()
    return any(pattern.search(lowered) is not None for pattern in _INJECTION_PATTERNS)


def select_observations(
    evidence: EvidencePackage, policy: QuestionPolicy
) -> list[ParsedObservation]:
    """Keep only relevant, non-empty observations for the approved field scope."""

    candidates = (
        [
            observation
            for observation in evidence.observations
            if observation.canonical_name in policy.requested_labs
        ]
        if policy.requested_labs
        else list(evidence.observations)
    )
    return [
        observation
        for observation in candidates
        if observation.status is not ObservationStatus.MISSING
    ]


def select_findings(
    evidence: EvidencePackage,
    *,
    observations: Sequence[ParsedObservation],
    policy: QuestionPolicy,
) -> list[LabFinding]:
    """Filter grouped findings to the selected lab scope."""

    allowed = {observation.canonical_name for observation in observations}
    if policy.requested_labs:
        allowed &= set(policy.requested_labs)
    return [finding for finding in evidence.findings if finding.canonical_name in allowed]


def select_conflicts(
    evidence: EvidencePackage,
    *,
    observations: Sequence[ParsedObservation],
    policy: QuestionPolicy,
) -> list[LabConflict]:
    """Filter conflicts to labs that remain in scope."""

    allowed = {observation.canonical_name for observation in observations}
    if policy.requested_labs:
        allowed &= set(policy.requested_labs)
    return [conflict for conflict in evidence.conflicts if conflict.canonical_name in allowed]


def select_missing_information(
    evidence: EvidencePackage,
    *,
    observations: Sequence[ParsedObservation],
    policy: QuestionPolicy,
) -> list[str]:
    """Report only missing information relevant to the question."""

    if not policy.requested_labs:
        return list(evidence.missing_information)
    observed = {observation.canonical_name for observation in observations}
    declared = {
        name for name in evidence.missing_information if name in policy.requested_labs
    }
    requested_missing = set(policy.requested_labs) - observed
    return sorted(declared | requested_missing)


class QuasiIdentifierReviewer:
    """Deterministic review flags for combinations that remain re-identifying."""

    def review(
        self,
        *,
        entity_types: frozenset[str],
        redaction_flags: Sequence[str],
        prompt_injection: bool,
        residual_pii: bool = False,
    ) -> list[str]:
        reasons: list[str] = []
        if "age_90_plus" in redaction_flags:
            reasons.append("age_90_plus")
        if "invalid_dob" in redaction_flags:
            reasons.append("invalid_dob")
        identifiers = _STRONG_IDENTIFIERS & entity_types
        if len(identifiers) >= 2:
            reasons.append("quasi_identifier_combination")
        if "FACILITY" in entity_types and "DATE_TIME" in entity_types:
            reasons.append("facility_and_date")
        if "FACILITY" in entity_types and (
            "AGE" in entity_types or "DATE_OF_BIRTH" in entity_types
        ):
            reasons.append("facility_and_age")
        if "PERSON" in entity_types and (
            "FACILITY" in entity_types or "DATE_TIME" in entity_types
        ):
            reasons.append("person_with_context")
        if prompt_injection:
            reasons.append("prompt_injection_suspected")
        if residual_pii:
            reasons.append("residual_pii")
        return list(dict.fromkeys(reasons))


def _requested_labs(lowered: str, aliases: Mapping[str, str]) -> frozenset[str]:
    found: set[str] = set()
    for alias in sorted(aliases, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered):
            found.add(aliases[alias])
    return frozenset(found)


def _has_term(lowered: str, terms: Sequence[str]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}", lowered) is not None for term in terms)

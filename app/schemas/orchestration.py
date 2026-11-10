"""Typed orchestration contracts: candidates, verdicts, urgent rules, and results.

Candidate and adjudication schemas validate external model output strictly.
Anything malformed, unsupported, or unsafe is rejected deterministically before
it can influence a final answer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.privacy import MinimizedConflict

REF_PATTERN = r"^\[[A-Z][A-Z0-9_]*_\d+\]$"


class OrchestrationStatus(StrEnum):
    """Typed terminal outcomes for one orchestration run."""

    VERIFIED = "verified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    VERIFICATION_FAILED = "verification_failed"
    PRIVACY_BLOCKED = "privacy_blocked"
    SAFETY_ESCALATION = "safety_escalation"
    EXTERNAL_UNAVAILABLE = "external_unavailable"


class VerifierName(StrEnum):
    """The mandatory deterministic verifiers applied to every claim."""

    SCHEMA = "schema"
    EVIDENCE = "evidence"
    NUMERIC = "numeric"
    TEMPORAL = "temporal"
    CONSISTENCY = "consistency"
    SAFETY = "safety"
    PROVENANCE = "provenance"


class UrgentRule(BaseModel):
    """A configured deterministic warning-sign rule over minimized observations.

    Rules are deployment configuration, never hardcoded clinical thresholds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1, max_length=64)
    lab_pattern: str = Field(min_length=1, max_length=128)
    operator: Literal["gt", "gte", "lt", "lte"]
    value: str = Field(pattern=r"^[+-]?\d+(?:\.\d+)?$")
    unit: str | None = Field(default=None, max_length=64)

    @field_validator("lab_pattern")
    @classmethod
    def require_valid_regex(cls, value: str) -> str:
        """Reject invalid patterns at configuration load instead of failing open."""

        import re

        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid lab_pattern regex: {exc}") from None
        return value


class CandidateClaim(BaseModel):
    """One claim parsed from an external candidate; refs use payload tokens."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=2000)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=16)
    confidence: Literal["low", "moderate"] | None = None

    @field_validator("evidence_refs")
    @classmethod
    def refs_must_be_token_shaped(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        import re

        pattern = re.compile(REF_PATTERN)
        for ref in value:
            if not pattern.match(ref):
                raise ValueError(f"evidence reference is not a payload token: {ref[:32]}")
        return value


class CandidateConflict(BaseModel):
    """A source conflict surfaced by a candidate, with tokenized references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lab: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=2000)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=16)


class ReasoningCandidate(BaseModel):
    """Strict schema for one reasoner's first-pass or revised output."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    facts: list[CandidateClaim] = Field(default_factory=list, max_length=32)
    hypotheses: list[CandidateClaim] = Field(default_factory=list, max_length=32)
    conflicts: list[CandidateConflict] = Field(default_factory=list, max_length=16)
    missing_information: list[str] = Field(default_factory=list, max_length=32)
    uncertainty: list[str] = Field(default_factory=list, max_length=32)
    review_notices: list[str] = Field(default_factory=list, max_length=32)

    @property
    def claims(self) -> list[CandidateClaim]:
        """Facts and hypotheses in one deterministic order."""

        return [*self.facts, *self.hypotheses]


class UnsupportedClaimNote(BaseModel):
    """A reviewer finding about one candidate claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=2000)
    reason: str = Field(min_length=1, max_length=2000)


class CritiqueOutput(BaseModel):
    """Strict schema for one reviewer's critique of one candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unsupported_claims: list[UnsupportedClaimNote] = Field(
        default_factory=list, max_length=32
    )
    numeric_or_unit_issues: list[str] = Field(default_factory=list, max_length=32)
    temporal_issues: list[str] = Field(default_factory=list, max_length=32)
    missing_citations: list[str] = Field(default_factory=list, max_length=32)
    safety_issues: list[str] = Field(default_factory=list, max_length=32)
    suggested_corrections: list[str] = Field(default_factory=list, max_length=32)


class AdjudicationCandidate(BaseModel):
    """Strict schema for the final adjudicator's structured synthesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verified_facts: list[CandidateClaim] = Field(default_factory=list, max_length=32)
    hypotheses: list[CandidateClaim] = Field(default_factory=list, max_length=32)
    unresolved_disagreements: list[str] = Field(default_factory=list, max_length=32)
    missing_information: list[str] = Field(default_factory=list, max_length=32)
    uncertainty: list[str] = Field(default_factory=list, max_length=32)
    review_notices: list[str] = Field(default_factory=list, max_length=32)


class VerifierResult(BaseModel):
    """Outcome of one verifier against one claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verifier: VerifierName
    passed: bool
    reason_codes: list[str] = Field(default_factory=list, max_length=16)
    matched_values: list[str] = Field(default_factory=list, max_length=16)


class ClaimVerdict(BaseModel):
    """Every verifier's outcome for one claim, plus the deterministic verdict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_text: str = Field(min_length=1, max_length=2000)
    evidence_refs: tuple[str, ...]
    results: tuple[VerifierResult, ...]
    is_fact: bool

    @property
    def passed(self) -> bool:
        """A claim passes only when every verifier passed."""

        return all(result.passed for result in self.results)


class ProviderUse(BaseModel):
    """Log-safe metadata for one routed provider call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot: str
    purpose: str
    provider: str | None = None
    model: str | None = None
    status: str
    error_code: str | None = None
    latency_ms: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class EvidenceCitation(BaseModel):
    """Resolved local provenance for one evidence reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ref: str
    document_id: UUID
    page_number: int
    chunk_id: str


class FinalClaim(BaseModel):
    """A claim in the user-facing answer with resolved citations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1, max_length=2000)
    evidence_refs: tuple[str, ...]
    citations: tuple[EvidenceCitation, ...] = Field(default_factory=tuple)
    confidence: Literal["low", "moderate"] | None = None


class OrchestrationResult(BaseModel):
    """The user-facing structured outcome. Hypotheses are never diagnoses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: OrchestrationStatus
    verified_facts: list[FinalClaim] = Field(default_factory=list, max_length=32)
    hypotheses: list[FinalClaim] = Field(default_factory=list, max_length=32)
    conflicts: list[MinimizedConflict] = Field(default_factory=list, max_length=16)
    missing_information: list[str] = Field(default_factory=list, max_length=32)
    uncertainty: list[str] = Field(default_factory=list, max_length=32)
    unresolved_disagreements: list[str] = Field(default_factory=list, max_length=32)
    review_notices: list[str] = Field(default_factory=list, max_length=32)
    clinician_review_required: bool = True
    abstain_reason: str | None = Field(default=None, max_length=128)
    reason_codes: list[str] = Field(default_factory=list, max_length=32)
    provider_usage: list[ProviderUse] = Field(default_factory=list, max_length=64)

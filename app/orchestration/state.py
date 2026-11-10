"""Typed LangGraph state for the bounded debate and verification workflow.

The state carries local-only references (evidence package, payload, token map).
Only validated, minimized content ever reaches an external provider, and audit
metadata is kept separately in ``provider_usage``.
"""

from __future__ import annotations

from typing import TypedDict
from uuid import UUID

from app.orchestration.context import VerificationContext
from app.schemas.evidence import EvidencePackage
from app.schemas.orchestration import (
    AdjudicationCandidate,
    ClaimVerdict,
    CritiqueOutput,
    OrchestrationResult,
    ProviderUse,
    ReasoningCandidate,
)
from app.schemas.privacy import PreparedPayload


class SlotCallOutcome(TypedDict, total=False):
    """Minimal routed-call summary for one provider slot.

    Audit fields (purpose, error codes, latency, tokens, provider/model) live in
    ``provider_usage``; the graph itself only needs the status and the text.
    """

    slot: str
    status: str
    text: str


class OrchestrationState(TypedDict, total=False):
    """Explicit, typed state; every transition is visible and bounded."""

    # Verified context
    request_id: str
    evidence: EvidencePackage
    payload: PreparedPayload
    verification_context: VerificationContext
    approvals: dict[str, UUID]

    # Safety precheck
    urgent_findings: list[str]

    # Independent reasoning
    reasoning_outcomes: dict[str, SlotCallOutcome]
    candidates: dict[str, ReasoningCandidate]
    malformed_slots: list[str]
    repair_attempted: bool

    # Deterministic validation
    verdicts: dict[str, list[ClaimVerdict]]
    valid_claims: dict[str, list[ClaimVerdict]]
    disagreement_labels: list[str]

    # Debate
    round: int
    critiques: dict[str, CritiqueOutput]
    sanitization_blocked: bool

    # Candidate gate
    gate_decision: str  # "adjudicate" | "insufficient" | "debate" | "blocked"

    # Final adjudication and validation
    adjudication_outcome: SlotCallOutcome | None
    adjudication: AdjudicationCandidate | None
    adjudication_verdicts: list[ClaimVerdict]
    adjudication_contradictions: list[str]
    repair_count: int

    # Budgets and result
    calls_made: int
    tokens_used: int
    provider_usage: list[ProviderUse]
    result: OrchestrationResult

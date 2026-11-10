"""The bounded LangGraph debate workflow with deterministic verification.

Nodes follow the approved plan: context loading, safety precheck, independent
reasoning, normalization, deterministic validation, critique, bounded revision,
safety review, candidate gate, final adjudication, final validation, and
repair/abstain. Every external call is routed through the approval-bound
provider router; nothing unvalidated can be presented as verified.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from typing import Any, cast
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from app.llm.errors import LLMProviderError
from app.llm.prompts import PromptKind, build_llm_request
from app.llm.router import ProviderRouter
from app.orchestration.context import (
    VerificationContext,
    build_verification_context,
    candidate_contradictions,
    verify_claim,
)
from app.orchestration.safety_rules import evaluate_urgent_rules
from app.orchestration.state import OrchestrationState, SlotCallOutcome
from app.orchestration.verifiers.schema import (
    SchemaError,
    parse_adjudication,
    parse_candidate,
    parse_critique,
)
from app.privacy.engine import PiiAnalysisError, PiiAnalyzer, PiiAnalyzerUnavailable
from app.privacy.redaction import TokenVault, redact_text
from app.schemas.orchestration import (
    AdjudicationCandidate,
    CandidateClaim,
    ClaimVerdict,
    CritiqueOutput,
    EvidenceCitation,
    FinalClaim,
    OrchestrationResult,
    OrchestrationStatus,
    ProviderUse,
    ReasoningCandidate,
    UrgentRule,
)
from app.schemas.privacy import DispatchPurpose
from app.schemas.providers import LLMErrorCode, ModelSlot

logger = logging.getLogger(__name__)

NodeFn = Callable[[OrchestrationState], Awaitable[dict[str, Any]]]
RouterFn = Callable[[OrchestrationState], str]

FINDING_LIMIT = 8
_FINALIZERS = (
    "finalize_urgent",
    "finalize_insufficient",
    "finalize_failed",
    "finalize_verified",
)


@dataclass(frozen=True)
class OrchestrationDependencies:
    """Injected services and bounds; the graph holds no global mutable state."""

    router: ProviderRouter
    reasoning_slots: tuple[str, ...]
    reviewer_slots: tuple[str, ...]
    adjudicator_slot: str
    max_debate_rounds: int
    max_provider_calls: int
    max_total_tokens: int
    max_output_tokens: int
    temperature: float
    urgent_rules: tuple[UrgentRule, ...]
    analyzer: PiiAnalyzer


def build_orchestration_graph(deps: OrchestrationDependencies) -> Any:
    """Compile the debate graph. All provider calls stay behind the router."""

    builder: StateGraph[OrchestrationState] = StateGraph(OrchestrationState)

    def register(name: str, fn: NodeFn) -> None:
        # LangGraph infers node types from direct function objects; the Callable
        # alias solves NodeInputT to Never, so registration goes through Any.
        builder.add_node(name, cast(Any, fn))

    register("load_verified_context", _load_verified_context())
    register("safety_precheck", _safety_precheck(deps))
    register("independent_reasoning", _independent_reasoning(deps))
    register("normalize_candidates", _normalize_candidates())
    register("repair_candidates", _repair_candidates(deps))
    register("deterministic_validation", _deterministic_validation())
    register("debate_critique", _debate_critique(deps))
    register("debate_revision", _debate_revision(deps))
    register("safety_review", _safety_review())
    register("candidate_gate", _candidate_gate(deps))
    register("final_adjudication", _final_adjudication(deps))
    register("final_validation", _final_validation())
    register("repair_or_abstain", _repair_or_abstain(deps))
    register("finalize_urgent", _finalize_urgent())
    register("finalize_insufficient", _finalize_insufficient())
    register("finalize_failed", _finalize_failed())
    register("finalize_verified", _finalize_verified())

    builder.add_edge(START, "load_verified_context")
    builder.add_conditional_edges(
        "load_verified_context",
        _route_after_load,
        {"continue": "safety_precheck", "insufficient": "finalize_insufficient"},
    )
    builder.add_conditional_edges(
        "safety_precheck",
        _route_after_precheck,
        {"continue": "independent_reasoning", "urgent": "finalize_urgent"},
    )
    builder.add_edge("independent_reasoning", "normalize_candidates")
    builder.add_conditional_edges(
        "normalize_candidates",
        _route_after_normalization,
        {
            "parsed": "deterministic_validation",
            "repair": "repair_candidates",
            "none": "finalize_failed",
        },
    )
    builder.add_conditional_edges(
        "repair_candidates",
        _route_after_repair,
        {"parsed": "deterministic_validation", "none": "finalize_failed"},
    )
    builder.add_conditional_edges(
        "deterministic_validation",
        _route_after_validation,
        {"has_valid": "debate_critique", "none": "finalize_insufficient"},
    )
    builder.add_edge("debate_critique", "debate_revision")
    builder.add_edge("debate_revision", "safety_review")
    builder.add_edge("safety_review", "candidate_gate")
    builder.add_conditional_edges(
        "candidate_gate",
        _route_after_gate,
        {
            "adjudicate": "final_adjudication",
            "debate": "debate_critique",
            "insufficient": "finalize_insufficient",
            "blocked": "finalize_failed",
        },
    )
    builder.add_edge("final_adjudication", "final_validation")
    builder.add_conditional_edges(
        "final_validation",
        _route_after_final_validation,
        {"verified": "finalize_verified", "repair": "repair_or_abstain"},
    )
    builder.add_conditional_edges(
        "repair_or_abstain",
        _route_after_repair_attempt,
        {"verified": "finalize_verified", "failed": "finalize_failed"},
    )
    for finalizer in _FINALIZERS:
        builder.add_edge(finalizer, END)
    return builder.compile()


# --------------------------------------------------------------------------
# Budget and call helpers
# --------------------------------------------------------------------------


def _approval_id(state: OrchestrationState, approval_key: str) -> UUID:
    approval_id = state["approvals"].get(approval_key)
    if approval_id is None:
        raise LLMProviderError(LLMErrorCode.APPROVAL_REQUIRED)
    return approval_id


async def _route_and_call(
    deps: OrchestrationDependencies,
    state: OrchestrationState,
    *,
    slot: str,
    purpose: DispatchPurpose,
    approval_key: str,
    request: Any,
) -> Any:
    """Dispatch one call against its own single-use, purpose-bound approval."""

    return await deps.router.generate(
        slot=ModelSlot(slot),
        request=request,
        approval_id=_approval_id(state, approval_key),
        purpose=purpose,
    )


def _round_number(state: OrchestrationState) -> int:
    """The 1-based debate round the current critique/revision calls belong to."""

    return state.get("round", 0) + 1


def _outcome(slot: str, result: Any) -> SlotCallOutcome:
    outcome = SlotCallOutcome(slot=slot, status=result.status.value)
    if result.response is not None:
        outcome["text"] = result.response.text
    return outcome


def _accumulate(
    entries: list[tuple[str, DispatchPurpose, Any]],
    calls_made: int,
    tokens_used: int,
    usage: list[ProviderUse],
) -> tuple[int, int, list[ProviderUse]]:
    """Fold routed-call results into the running budget counters."""

    usage = list(usage)
    for slot, purpose, result in entries:
        usage.append(
            ProviderUse(
                slot=slot,
                purpose=purpose.value,
                provider=result.destination.provider if result.destination else None,
                model=result.destination.model if result.destination else None,
                status=result.status.value,
                error_code=result.error_code.value if result.error_code else None,
                latency_ms=result.response.latency_ms if result.response else 0,
                total_tokens=result.response.usage.total_tokens if result.response else 0,
            )
        )
        if result.status.value in {"completed", "failed"}:
            calls_made += 1
            tokens_used += result.response.usage.total_tokens if result.response else 0
    return calls_made, tokens_used, usage


def _budget_update(
    calls_made: int,
    tokens_used: int,
    usage: list[ProviderUse],
) -> dict[str, Any]:
    return {
        "calls_made": calls_made,
        "tokens_used": tokens_used,
        "provider_usage": usage,
    }


def _calls_remaining(state: OrchestrationState, deps: OrchestrationDependencies) -> int:
    return max(0, deps.max_provider_calls - state.get("calls_made", 0))


def _tokens_remaining(state: OrchestrationState, deps: OrchestrationDependencies) -> int:
    return max(0, deps.max_total_tokens - state.get("tokens_used", 0))


async def _sanitize_texts(
    deps: OrchestrationDependencies, texts: list[str], today: date
) -> list[str] | None:
    """Sanitize model-to-model debate material; None means fail closed.

    Presidio analysis runs in a worker thread so the event loop is never blocked.
    """

    vault = TokenVault()
    sanitized: list[str] = []
    try:
        for text in texts:
            result = await asyncio.to_thread(
                redact_text, text, analyzer=deps.analyzer, vault=vault, on_date=today
            )
            sanitized.append(result.text)
    except (PiiAnalyzerUnavailable, PiiAnalysisError):
        logger.warning("debate sanitization unavailable; blocking further debate calls")
        return None
    except Exception:
        logger.warning("debate sanitization failed; blocking further debate calls")
        return None
    return sanitized


def _validated_texts_by_slot(state: OrchestrationState) -> dict[str, list[str]]:
    """Validated claim texts per slot, in verdict order."""

    by_slot: dict[str, list[str]] = {}
    for slot, verdicts in state.get("valid_claims", {}).items():
        texts = [verdict.claim_text for verdict in verdicts]
        if texts:
            by_slot[slot] = texts
    return by_slot


async def _sanitized_by_slot(
    state: OrchestrationState, deps: OrchestrationDependencies
) -> dict[str, list[str]] | None:
    """Sanitized validated claims per slot, or None when sanitization fails."""

    by_slot = _validated_texts_by_slot(state)
    flat = [text for slot in by_slot for text in by_slot[slot]]
    if not flat:
        return {}
    sanitized = await _sanitize_texts(deps, flat, date.today())
    if sanitized is None:
        return None
    result: dict[str, list[str]] = {}
    cursor = 0
    for slot, texts in by_slot.items():
        result[slot] = sanitized[cursor : cursor + len(texts)]
        cursor += len(texts)
    return result


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def _load_verified_context() -> NodeFn:
    async def node(state: OrchestrationState) -> dict[str, Any]:
        context = build_verification_context(state["payload"], state["evidence"])
        return {"verification_context": context}

    return node


def _safety_precheck(deps: OrchestrationDependencies) -> NodeFn:
    async def node(state: OrchestrationState) -> dict[str, Any]:
        findings = evaluate_urgent_rules(state["payload"], deps.urgent_rules)
        return {"urgent_findings": [finding.label for finding in findings]}

    return node


def _route_after_load(state: OrchestrationState) -> str:
    if not state["payload"].observations:
        return "insufficient"
    return "continue"


def _route_after_precheck(state: OrchestrationState) -> str:
    if state.get("urgent_findings"):
        return "urgent"
    return "continue"


def _independent_reasoning(deps: OrchestrationDependencies) -> NodeFn:
    """Parallel first-pass calls: identical request, no peer output, budget-capped."""

    async def node(state: OrchestrationState) -> dict[str, Any]:
        remaining = _calls_remaining(state, deps)
        tokens_left = _tokens_remaining(state, deps)
        available = list(deps.reasoning_slots[:remaining]) if tokens_left > 0 else []
        if not available:
            return {}
        request = build_llm_request(
            kind=PromptKind.REASONING,
            payload=state["payload"],
            request_id=state["request_id"],
            max_output_tokens=deps.max_output_tokens,
            temperature=deps.temperature,
        )
        results = await asyncio.gather(
            *[
                _route_and_call(
                    deps,
                    state,
                    slot=slot,
                    purpose=DispatchPurpose.REASONING,
                    approval_key=f"{slot}:reasoning",
                    request=request,
                )
                for slot in available
            ]
        )
        outcomes = {
            slot: _outcome(slot, result)
            for slot, result in zip(available, results, strict=False)
        }
        entries = [
            (slot, DispatchPurpose.REASONING, result)
            for slot, result in zip(available, results, strict=False)
        ]
        calls, tokens, usage = _accumulate(
            entries,
            state.get("calls_made", 0),
            state.get("tokens_used", 0),
            state.get("provider_usage", []),
        )
        updates: dict[str, Any] = {"reasoning_outcomes": outcomes}
        updates.update(_budget_update(calls, tokens, usage))
        return updates

    return node


def _parse_outcomes(
    state: OrchestrationState,
) -> tuple[dict[str, ReasoningCandidate], list[str]]:
    candidates: dict[str, ReasoningCandidate] = {}
    malformed: list[str] = []
    for slot, outcome in state.get("reasoning_outcomes", {}).items():
        text = outcome.get("text")
        if outcome.get("status") != "completed" or not text:
            continue
        try:
            candidates[slot] = parse_candidate(text)
        except SchemaError:
            malformed.append(slot)
    return candidates, malformed


def _normalize_candidates() -> NodeFn:
    async def node(state: OrchestrationState) -> dict[str, Any]:
        candidates, malformed = _parse_outcomes(state)
        return {
            "candidates": candidates,
            "malformed_slots": malformed,
            "repair_attempted": False,
        }

    return node


def _route_after_normalization(state: OrchestrationState) -> str:
    if state.get("malformed_slots") and not state.get("repair_attempted"):
        return "repair"
    return "parsed" if state.get("candidates") else "none"


def _repair_candidates(deps: OrchestrationDependencies) -> NodeFn:
    """One bounded schema re-ask for malformed candidates; raw output is never re-sent."""

    async def node(state: OrchestrationState) -> dict[str, Any]:
        malformed = state.get("malformed_slots", [])
        request = build_llm_request(
            kind=PromptKind.REVISION,
            payload=state["payload"],
            request_id=state["request_id"],
            max_output_tokens=deps.max_output_tokens,
            temperature=deps.temperature,
            discussion=json.dumps(
                {
                    "problem": "The previous output was not a schema-valid JSON object.",
                    "instruction": (
                        "Return exactly one JSON object matching the required schema. "
                        "Do not include commentary or markdown fences."
                    ),
                },
                sort_keys=True,
            ),
        )
        calls_made = state.get("calls_made", 0)
        tokens_used = state.get("tokens_used", 0)
        usage = state.get("provider_usage", [])
        repaired: dict[str, ReasoningCandidate] = dict(state.get("candidates", {}))
        still_malformed: list[str] = []
        entries: list[tuple[str, DispatchPurpose, Any]] = []
        for slot in malformed:
            result = await _route_and_call(
                deps,
                state,
                slot=slot,
                purpose=DispatchPurpose.REVISION,
                approval_key=f"{slot}:revision:malformed",
                request=request,
            )
            entries.append((slot, DispatchPurpose.REVISION, result))
            outcome = _outcome(slot, result)
            if outcome.get("status") == "completed" and outcome.get("text"):
                try:
                    repaired[slot] = parse_candidate(outcome["text"])
                    continue
                except SchemaError:
                    logger.warning("candidate repair for slot=%s stayed malformed", slot)
            still_malformed.append(slot)
        calls_made, tokens_used, usage = _accumulate(
            entries, calls_made, tokens_used, usage
        )
        updates: dict[str, Any] = {
            "candidates": repaired,
            "malformed_slots": still_malformed,
            "repair_attempted": True,
        }
        updates.update(_budget_update(calls_made, tokens_used, usage))
        return updates

    return node


def _route_after_repair(state: OrchestrationState) -> str:
    return "parsed" if state.get("candidates") else "none"


def _validate_all(state: OrchestrationState) -> dict[str, Any]:
    context: VerificationContext = state["verification_context"]
    verdicts: dict[str, list[ClaimVerdict]] = {}
    contradictions: dict[str, list[str]] = {}
    valid_claims: dict[str, list[ClaimVerdict]] = {}
    for slot, candidate in state.get("candidates", {}).items():
        candidate_contradiction = candidate_contradictions(candidate, context)
        contradictions[slot] = candidate_contradiction
        claim_verdicts = [
            verify_claim(claim=claim, candidate=candidate, context=context, is_fact=True)
            for claim in candidate.facts
        ] + [
            verify_claim(claim=claim, candidate=candidate, context=context, is_fact=False)
            for claim in candidate.hypotheses
        ]
        verdicts[slot] = claim_verdicts
        valid_claims[slot] = (
            [] if candidate_contradiction else [v for v in claim_verdicts if v.passed]
        )
    disagreement_labels = _cross_candidate_disagreements(valid_claims, context)
    return {
        "verdicts": verdicts,
        "valid_claims": valid_claims,
        "disagreement_labels": disagreement_labels,
    }


def _cross_candidate_disagreements(
    valid_claims: dict[str, list[ClaimVerdict]], context: VerificationContext
) -> list[str]:
    merged = ReasoningCandidate(
        facts=[
            CandidateClaim(text=verdict.claim_text, evidence_refs=verdict.evidence_refs)
            for verdicts in valid_claims.values()
            for verdict in verdicts
            if verdict.is_fact
        ],
        hypotheses=[
            CandidateClaim(
                text=verdict.claim_text,
                evidence_refs=verdict.evidence_refs,
                confidence="low",
            )
            for verdicts in valid_claims.values()
            for verdict in verdicts
            if not verdict.is_fact
        ],
    )
    return candidate_contradictions(merged, context)


def _deterministic_validation() -> NodeFn:
    async def node(state: OrchestrationState) -> dict[str, Any]:
        return _validate_all(state)

    return node


def _route_after_validation(state: OrchestrationState) -> str:
    if any(state.get("valid_claims", {}).values()):
        return "has_valid"
    return "none"


def _deterministic_findings(state: OrchestrationState) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for slot, slot_verdicts in state.get("verdicts", {}).items():
        failures = [
            {
                "reasons": sorted(
                    {
                        f"{result.verifier.value}:{code}"
                        for result in verdict.results
                        if not result.passed
                        for code in result.reason_codes
                    }
                )
            }
            for verdict in slot_verdicts
            if not verdict.passed
        ][:FINDING_LIMIT]
        if failures:
            findings.append({"slot": slot, "rejected_claim_findings": failures})
    return findings


def _critique_findings(state: OrchestrationState) -> list[dict[str, Any]]:
    return [
        {
            "slot": slot,
            "unsupported_claims": [
                f"{note.text}: {note.reason}" for note in critique.unsupported_claims
            ],
            "safety_issues": critique.safety_issues,
            "numeric_or_unit_issues": critique.numeric_or_unit_issues,
            "temporal_issues": critique.temporal_issues,
            "missing_citations": critique.missing_citations,
        }
        for slot, critique in state.get("critiques", {}).items()
    ]


def _debate_critique(deps: OrchestrationDependencies) -> NodeFn:
    async def node(state: OrchestrationState) -> dict[str, Any]:
        reviewers = deps.reviewer_slots
        if (
            not reviewers
            or state.get("round", 0) >= deps.max_debate_rounds
            or len(reviewers) > _calls_remaining(state, deps)
            or _tokens_remaining(state, deps) <= 0
        ):
            return {}
        sanitized = await _sanitized_by_slot(state, deps)
        if sanitized is None:
            return {"sanitization_blocked": True}
        request = build_llm_request(
            kind=PromptKind.CRITIQUE,
            payload=state["payload"],
            request_id=state["request_id"],
            max_output_tokens=deps.max_output_tokens,
            temperature=deps.temperature,
            discussion=json.dumps(
                {
                    "candidate_claims": [
                        {"slot": slot, "claims": texts} for slot, texts in sanitized.items()
                    ],
                    "reviewer_findings": _critique_findings(state),
                    "disagreements": state.get("disagreement_labels", []),
                },
                sort_keys=True,
            ),
        )
        results = await asyncio.gather(
            *[
                _route_and_call(
                    deps,
                    state,
                    slot=slot,
                    purpose=DispatchPurpose.CRITIQUE,
                    approval_key=f"{slot}:critique:r{_round_number(state)}",
                    request=request,
                )
                for slot in reviewers
            ]
        )
        entries = [
            (slot, DispatchPurpose.CRITIQUE, result)
            for slot, result in zip(reviewers, results, strict=False)
        ]
        critiques: dict[str, CritiqueOutput] = {}
        outcomes: dict[str, SlotCallOutcome] = {}
        for slot, result in zip(reviewers, results, strict=False):
            outcome = _outcome(slot, result)
            outcomes[slot] = outcome
            if outcome.get("status") == "completed" and outcome.get("text"):
                try:
                    critiques[slot] = parse_critique(outcome["text"])
                except SchemaError:
                    logger.warning("critique from slot=%s was schema-invalid; ignored", slot)
        calls, tokens, usage = _accumulate(
            entries,
            state.get("calls_made", 0),
            state.get("tokens_used", 0),
            state.get("provider_usage", []),
        )
        updates: dict[str, Any] = {
            "critiques": {**state.get("critiques", {}), **critiques},
        }
        updates.update(_budget_update(calls, tokens, usage))
        return updates

    return node


def _has_critique_findings(state: OrchestrationState) -> bool:
    return any(
        critique.unsupported_claims
        or critique.suggested_corrections
        or critique.safety_issues
        or critique.numeric_or_unit_issues
        or critique.temporal_issues
        or critique.missing_citations
        for critique in state.get("critiques", {}).values()
    )


def _revision_targets(state: OrchestrationState) -> list[str]:
    """Slots whose candidates have rejected claims or face reviewer findings."""

    critiques_present = _has_critique_findings(state)
    targets: list[str] = []
    for slot, verdicts in state.get("verdicts", {}).items():
        if slot not in state.get("candidates", {}):
            continue
        if any(not verdict.passed for verdict in verdicts) or critiques_present:
            targets.append(slot)
    return targets


def _debate_revision(deps: OrchestrationDependencies) -> NodeFn:
    """One bounded revision round; a revised candidate replaces the original only if valid."""

    async def node(state: OrchestrationState) -> dict[str, Any]:
        next_round = state.get("round", 0) + 1
        if state.get("sanitization_blocked"):
            return {"round": next_round}
        targets = _revision_targets(state)
        if (
            not targets
            or len(targets) > _calls_remaining(state, deps)
            or _tokens_remaining(state, deps) <= 0
        ):
            return {"round": next_round}
        sanitized = await _sanitized_by_slot(state, deps)
        if sanitized is None:
            return {"round": next_round, "sanitization_blocked": True}
        deterministic = _deterministic_findings(state)
        reviewer_findings = _critique_findings(state)
        calls_made = state.get("calls_made", 0)
        tokens_used = state.get("tokens_used", 0)
        usage = state.get("provider_usage", [])
        candidates = dict(state.get("candidates", {}))
        outcomes: dict[str, SlotCallOutcome] = {}
        entries: list[tuple[str, DispatchPurpose, Any]] = []
        for slot in targets:
            discussion = json.dumps(
                {
                    "previous_candidate_claims": sanitized.get(slot, []),
                    "reviewer_findings": reviewer_findings,
                    "deterministic_findings": deterministic,
                },
                sort_keys=True,
            )
            request = build_llm_request(
                kind=PromptKind.REVISION,
                payload=state["payload"],
                request_id=state["request_id"],
                max_output_tokens=deps.max_output_tokens,
                temperature=deps.temperature,
                discussion=discussion,
            )
            result = await _route_and_call(
                deps,
                state,
                slot=slot,
                purpose=DispatchPurpose.REVISION,
                approval_key=f"{slot}:revision:r{_round_number(state)}",
                request=request,
            )
            entries.append((slot, DispatchPurpose.REVISION, result))
            outcome = _outcome(slot, result)
            outcomes[slot] = outcome
            if outcome.get("status") == "completed" and outcome.get("text"):
                try:
                    candidates[slot] = parse_candidate(outcome["text"])
                except SchemaError:
                    logger.warning("revision from slot=%s was schema-invalid; kept prior", slot)
        calls_made, tokens_used, usage = _accumulate(entries, calls_made, tokens_used, usage)
        updates: dict[str, Any] = {
            "candidates": candidates,
            "round": next_round,
        }
        updates.update(_budget_update(calls_made, tokens_used, usage))
        return updates

    return node


def _safety_review() -> NodeFn:
    """Safety takes precedence over scores and votes: full validation is re-run."""

    async def node(state: OrchestrationState) -> dict[str, Any]:
        return _validate_all(state)

    return node


def _candidate_gate(deps: OrchestrationDependencies) -> NodeFn:
    async def node(state: OrchestrationState) -> dict[str, Any]:
        if state.get("sanitization_blocked"):
            return {"gate_decision": "blocked"}
        if not any(state.get("valid_claims", {}).values()):
            return {"gate_decision": "insufficient"}
        can_debate = (
            bool(deps.reviewer_slots)
            and state.get("round", 0) < deps.max_debate_rounds
            and bool(state.get("disagreement_labels"))
            and len(deps.reviewer_slots) <= _calls_remaining(state, deps)
            and _tokens_remaining(state, deps) > 0
        )
        if can_debate:
            return {"gate_decision": "debate"}
        return {"gate_decision": "adjudicate"}

    return node


def _route_after_gate(state: OrchestrationState) -> str:
    return state.get("gate_decision", "insufficient")


async def _adjudication_discussion(
    state: OrchestrationState, deps: OrchestrationDependencies
) -> str | None:
    """Only validated claims, disagreements, and uncertainty reach the adjudicator."""

    sanitized = await _sanitized_by_slot(state, deps)
    if sanitized is None:
        return None
    if state.get("valid_claims") and not sanitized:
        return None
    validated: list[dict[str, Any]] = []
    for slot, verdicts in state.get("valid_claims", {}).items():
        texts = sanitized.get(slot, [])
        if not texts:
            continue
        validated.append(
            {
                "slot": slot,
                "claims": [
                    {"text": text, "is_fact": verdict.is_fact}
                    for text, verdict in zip(texts, verdicts, strict=False)
                ],
            }
        )
    uncertainty = sorted(
        {
            notice
            for candidate in state.get("candidates", {}).values()
            for notice in candidate.uncertainty
        }
    )
    missing = sorted(
        {
            item
            for candidate in state.get("candidates", {}).values()
            for item in candidate.missing_information
        }
    )
    return json.dumps(
        {
            "validated_claims": validated,
            "disagreements": state.get("disagreement_labels", []),
            "uncertainty": uncertainty,
            "missing_information": missing,
        },
        sort_keys=True,
    )


def _final_adjudication(deps: OrchestrationDependencies) -> NodeFn:
    async def node(state: OrchestrationState) -> dict[str, Any]:
        if _calls_remaining(state, deps) < 1 or _tokens_remaining(state, deps) <= 0:
            return {}
        discussion = await _adjudication_discussion(state, deps)
        if discussion is None:
            return {"sanitization_blocked": True}
        request = build_llm_request(
            kind=PromptKind.ADJUDICATION,
            payload=state["payload"],
            request_id=state["request_id"],
            max_output_tokens=deps.max_output_tokens,
            temperature=deps.temperature,
            discussion=discussion,
        )
        slot = deps.adjudicator_slot
        result = await _route_and_call(
            deps,
            state,
            slot=slot,
            purpose=DispatchPurpose.ADJUDICATION,
            approval_key=f"{slot}:adjudication",
            request=request,
        )
        outcome = _outcome(slot, result)
        updates: dict[str, Any] = {"adjudication_outcome": outcome}
        if outcome.get("status") == "completed" and outcome.get("text"):
            try:
                updates["adjudication"] = parse_adjudication(outcome["text"])
            except SchemaError:
                logger.warning("adjudication output was schema-invalid; repair may follow")
        calls, tokens, usage = _accumulate(
            [(slot, DispatchPurpose.ADJUDICATION, result)],
            state.get("calls_made", 0),
            state.get("tokens_used", 0),
            state.get("provider_usage", []),
        )
        updates.update(_budget_update(calls, tokens, usage))
        return updates

    return node


def _as_reasoning_candidate(adjudication: AdjudicationCandidate) -> ReasoningCandidate:
    return ReasoningCandidate(
        facts=[
            CandidateClaim(
                text=claim.text, evidence_refs=claim.evidence_refs, confidence=claim.confidence
            )
            for claim in adjudication.verified_facts
        ],
        hypotheses=[
            CandidateClaim(
                text=claim.text, evidence_refs=claim.evidence_refs, confidence=claim.confidence
            )
            for claim in adjudication.hypotheses
        ],
    )


def _verify_adjudication(
    state: OrchestrationState, adjudication: AdjudicationCandidate
) -> dict[str, Any]:
    context: VerificationContext = state["verification_context"]
    wrapped = _as_reasoning_candidate(adjudication)
    verdicts = [
        verify_claim(claim=claim, candidate=wrapped, context=context, is_fact=True)
        for claim in wrapped.facts
    ] + [
        verify_claim(claim=claim, candidate=wrapped, context=context, is_fact=False)
        for claim in wrapped.hypotheses
    ]
    return {
        "adjudication_verdicts": verdicts,
        "adjudication_contradictions": candidate_contradictions(wrapped, context),
    }


def _final_validation() -> NodeFn:
    async def node(state: OrchestrationState) -> dict[str, Any]:
        adjudication = state.get("adjudication")
        if adjudication is None:
            return {
                "adjudication_verdicts": [],
                "adjudication_contradictions": ["schema_invalid"],
            }
        return _verify_adjudication(state, adjudication)

    return node


def _adjudication_verdict(state: OrchestrationState) -> str:
    """Classify the adjudication: "verified" or "unverified"."""

    if state.get("adjudication") is None:
        return "unverified"
    if state.get("adjudication_contradictions"):
        return "unverified"
    verdicts = state.get("adjudication_verdicts", [])
    if verdicts and all(verdict.passed for verdict in verdicts):
        return "verified"
    return "unverified"


def _route_after_final_validation(state: OrchestrationState) -> str:
    return "verified" if _adjudication_verdict(state) == "verified" else "repair"


def _adjudication_failure_summary(state: OrchestrationState) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not state.get("adjudication"):
        findings.append({"issue": "schema_invalid"})
    for verdict in state.get("adjudication_verdicts", []):
        if verdict.passed:
            continue
        findings.append(
            {
                "issue": "claim_verification_failed",
                "reasons": sorted(
                    {
                        f"{result.verifier.value}:{code}"
                        for result in verdict.results
                        if not result.passed
                        for code in result.reason_codes
                    }
                ),
            }
        )
    for contradiction in state.get("adjudication_contradictions", []):
        findings.append({"issue": "internal_contradiction", "detail": contradiction})
    return findings[:FINDING_LIMIT]


def _repair_or_abstain(deps: OrchestrationDependencies) -> NodeFn:
    """At most one repair for a repairable defect; otherwise typed abstention."""

    async def node(state: OrchestrationState) -> dict[str, Any]:
        if (
            state.get("repair_count", 0) >= 1
            or state.get("sanitization_blocked")
            or _calls_remaining(state, deps) < 1
            or _tokens_remaining(state, deps) <= 0
        ):
            return {"repair_count": 1}
        request = build_llm_request(
            kind=PromptKind.ADJUDICATION,
            payload=state["payload"],
            request_id=state["request_id"],
            max_output_tokens=deps.max_output_tokens,
            temperature=deps.temperature,
            discussion=json.dumps(
                {
                    "problem": "The previous synthesis failed deterministic verification.",
                    "findings": _adjudication_failure_summary(state),
                    "instruction": (
                        "Return one corrected JSON object. Remove unsupported claims; "
                        "never invent evidence references."
                    ),
                },
                sort_keys=True,
            ),
        )
        slot = deps.adjudicator_slot
        result = await _route_and_call(
            deps,
            state,
            slot=slot,
            purpose=DispatchPurpose.REPAIR,
            approval_key=f"{slot}:repair",
            request=request,
        )
        outcome = _outcome(slot, result)
        updates: dict[str, Any] = {"repair_count": 1, "adjudication_outcome": outcome}
        if outcome.get("status") == "completed" and outcome.get("text"):
            try:
                candidate = parse_adjudication(outcome["text"])
                updates["adjudication"] = candidate
                updates.update(_verify_adjudication(state, candidate))
            except SchemaError:
                logger.warning("repair output remained schema-invalid")
        calls, tokens, usage = _accumulate(
            [(slot, DispatchPurpose.REPAIR, result)],
            state.get("calls_made", 0),
            state.get("tokens_used", 0),
            state.get("provider_usage", []),
        )
        updates.update(_budget_update(calls, tokens, usage))
        return updates

    return node


def _route_after_repair_attempt(state: OrchestrationState) -> str:
    return "verified" if _adjudication_verdict(state) == "verified" else "failed"


# --------------------------------------------------------------------------
# Finalizers
# --------------------------------------------------------------------------


def _citations_for(
    evidence_refs: tuple[str, ...], context: VerificationContext
) -> tuple[EvidenceCitation, ...]:
    citations: list[EvidenceCitation] = []
    seen: set[tuple[str, int, str]] = set()
    for ref in evidence_refs:
        for evidence_id in context.token_map.get(ref, frozenset()):
            source = context.evidence.resolve(evidence_id)
            if source is None:
                continue
            key = (str(source.document_id), source.page_number, source.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                EvidenceCitation(
                    ref=ref,
                    document_id=source.document_id,
                    page_number=source.page_number,
                    chunk_id=source.chunk_id,
                )
            )
    return tuple(citations)


def _finalize_urgent() -> NodeFn:
    async def node(state: OrchestrationState) -> dict[str, Any]:
        return {
            "result": OrchestrationResult(
                status=OrchestrationStatus.SAFETY_ESCALATION,
                review_notices=[
                    "urgent_warning_signs_bypassed_debate",
                    *state.get("urgent_findings", []),
                ],
                clinician_review_required=True,
                provider_usage=state.get("provider_usage", []),
            )
        }

    return node


def _finalize_insufficient() -> NodeFn:
    async def node(state: OrchestrationState) -> dict[str, Any]:
        payload = state["payload"]
        reason = "no_observations" if not payload.observations else "no_validated_claims"
        return {
            "result": OrchestrationResult(
                status=OrchestrationStatus.INSUFFICIENT_EVIDENCE,
                conflicts=payload.conflicts,
                missing_information=list(payload.missing_information),
                abstain_reason=reason,
                clinician_review_required=True,
                provider_usage=state.get("provider_usage", []),
            )
        }

    return node


def _finalize_failed() -> NodeFn:
    async def node(state: OrchestrationState) -> dict[str, Any]:
        reasons: list[str] = []
        if state.get("sanitization_blocked"):
            reasons.append("sanitization_unavailable")
        if not state.get("candidates"):
            reasons.append("all_candidates_malformed")
        outcome = state.get("adjudication_outcome")
        if outcome is not None and outcome.get("status") != "completed":
            reasons.append("adjudication_call_failed")
        if state.get("repair_count", 0) >= 1:
            reasons.append("repair_exhausted")
        if not reasons:
            reasons.append("final_verification_failed")
        return {
            "result": OrchestrationResult(
                status=OrchestrationStatus.VERIFICATION_FAILED,
                abstain_reason=reasons[0],
                reason_codes=reasons,
                review_notices=["verification_failed_abstaining"],
                clinician_review_required=True,
                provider_usage=state.get("provider_usage", []),
            )
        }

    return node


def _finalize_verified() -> NodeFn:
    async def node(state: OrchestrationState) -> dict[str, Any]:
        context: VerificationContext = state["verification_context"]
        adjudication = state.get("adjudication")
        if adjudication is None:  # unreachable via the verified route; fail closed
            return {
                "result": OrchestrationResult(
                    status=OrchestrationStatus.VERIFICATION_FAILED,
                    abstain_reason="final_state_missing",
                )
            }
        payload = state["payload"]
        facts = [
            FinalClaim(
                text=claim.text,
                evidence_refs=claim.evidence_refs,
                citations=_citations_for(claim.evidence_refs, context),
                confidence=claim.confidence,
            )
            for claim in adjudication.verified_facts
        ]
        hypotheses = [
            FinalClaim(
                text=claim.text,
                evidence_refs=claim.evidence_refs,
                citations=_citations_for(claim.evidence_refs, context),
                confidence=claim.confidence,
            )
            for claim in adjudication.hypotheses
        ]
        return {
            "result": OrchestrationResult(
                status=OrchestrationStatus.VERIFIED,
                verified_facts=facts,
                hypotheses=hypotheses,
                conflicts=payload.conflicts,
                missing_information=sorted(
                    {*adjudication.missing_information, *payload.missing_information}
                ),
                uncertainty=sorted({*adjudication.uncertainty}),
                unresolved_disagreements=sorted(
                    {
                        *adjudication.unresolved_disagreements,
                        *state.get("disagreement_labels", []),
                    }
                ),
                review_notices=sorted({*adjudication.review_notices}),
                clinician_review_required=True,
                provider_usage=state.get("provider_usage", []),
            )
        }

    return node

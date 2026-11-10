"""Entry service: destination-specific approvals, graph execution, typed results."""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from app.core.config import Settings
from app.llm.config import SlotConfig
from app.llm.router import ProviderRouter
from app.orchestration.graph import OrchestrationDependencies, build_orchestration_graph
from app.privacy.engine import PiiAnalyzer
from app.privacy.gate import PrivacyGate
from app.schemas.evidence import EvidencePackage
from app.schemas.orchestration import OrchestrationResult, OrchestrationStatus
from app.schemas.privacy import DispatchPurpose, PrivacyStatus
from app.schemas.providers import ModelSlot

logger = logging.getLogger(__name__)



class OrchestrationService:
    """Run the debate graph only when every destination is individually approved."""

    def __init__(
        self,
        *,
        gate: PrivacyGate,
        deps: OrchestrationDependencies,
        slots: dict[str, SlotConfig],
        timeout_seconds: float,
    ) -> None:
        self._gate = gate
        self._slots = slots
        self._deps = deps
        self._graph = build_orchestration_graph(self._deps)
        self._timeout_seconds = timeout_seconds

    async def run(
        self,
        *,
        evidence: EvidencePackage,
        question: str,
        request_id: str,
    ) -> OrchestrationResult:
        """Approve every destination, then execute the graph; never fail open."""

        if not self._deps.reasoning_slots:
            return OrchestrationResult(
                status=OrchestrationStatus.EXTERNAL_UNAVAILABLE,
                abstain_reason="no_reasoning_slots_configured",
                review_notices=["external_reasoning_unavailable"],
            )
        missing = [
            slot
            for slot in (*self._deps.reasoning_slots, self._deps.adjudicator_slot)
            if slot not in self._slots
        ]
        if missing:
            return OrchestrationResult(
                status=OrchestrationStatus.EXTERNAL_UNAVAILABLE,
                abstain_reason="required_slot_not_configured",
                reason_codes=missing[:32],
                review_notices=["external_reasoning_unavailable"],
            )

        destinations = _required_destinations(self._deps, slots=self._slots)
        approvals: dict[str, UUID] = {}
        unapproved: list[str] = []
        payload = None
        for key, config in destinations.items():
            decision = await self._gate.prepare(
                evidence=evidence,
                question=question,
                destination=config.destination(),
                request_id=request_id,
            )
            if decision.status is PrivacyStatus.APPROVED and decision.payload is not None:
                approvals[key] = decision.approval_id
                payload = payload or decision.payload
            else:
                unapproved.append(f"{key}:{decision.status.value}")
        if unapproved or payload is None:
            logger.warning(
                "orchestration blocked request_id=%s unapproved=%d", request_id, len(unapproved)
            )
            return OrchestrationResult(
                status=OrchestrationStatus.PRIVACY_BLOCKED,
                abstain_reason="destination_approval_missing",
                reason_codes=unapproved[:32],
                review_notices=["no_external_call_was_made"],
            )

        state: dict[str, object] = {
            "request_id": request_id,
            "evidence": evidence,
            "payload": payload,
            "approvals": approvals,
            "calls_made": 0,
            "tokens_used": 0,
            "provider_usage": [],
            "round": 0,
            "repair_count": 0,
        }
        try:
            async with asyncio.timeout(self._timeout_seconds):
                final_state = await self._graph.ainvoke(state)
        except TimeoutError:
            logger.warning("orchestration timeout request_id=%s", request_id)
            return OrchestrationResult(
                status=OrchestrationStatus.VERIFICATION_FAILED,
                abstain_reason="orchestration_timeout",
                review_notices=["orchestration_timeout_abstaining"],
                clinician_review_required=True,
            )
        result = final_state.get("result")
        if isinstance(result, OrchestrationResult):
            return result
        return OrchestrationResult(
            status=OrchestrationStatus.VERIFICATION_FAILED,
            abstain_reason="orchestration_failed",
            review_notices=["orchestration_failed_abstaining"],
        )


def _required_destinations(
    deps: OrchestrationDependencies, *, slots: dict[str, SlotConfig]
) -> dict[str, SlotConfig]:
    """Every (slot, purpose, call) the graph may dispatch, with its exact destination.

    Approvals are single-use, so each anticipated provider call gets its own
    destination-bound approval: one per reasoner for reasoning and the malformed
    re-ask, one per reviewer per debate round for critique, one per reasoner per
    round for revision, and separate adjudication and repair approvals.
    """

    required: dict[str, SlotConfig] = {}
    for slot in deps.reasoning_slots:
        required[f"{slot}:reasoning"] = _with_purpose(slots[slot], DispatchPurpose.REASONING)
        required[f"{slot}:revision:malformed"] = _with_purpose(
            slots[slot], DispatchPurpose.REVISION
        )
        for round_number in range(1, deps.max_debate_rounds + 1):
            required[f"{slot}:revision:r{round_number}"] = _with_purpose(
                slots[slot], DispatchPurpose.REVISION
            )
    for slot in deps.reviewer_slots:
        for round_number in range(1, deps.max_debate_rounds + 1):
            required[f"{slot}:critique:r{round_number}"] = _with_purpose(
                slots[slot], DispatchPurpose.CRITIQUE
            )
    required[f"{deps.adjudicator_slot}:adjudication"] = _with_purpose(
        slots[deps.adjudicator_slot], DispatchPurpose.ADJUDICATION
    )
    required[f"{deps.adjudicator_slot}:repair"] = _with_purpose(
        slots[deps.adjudicator_slot], DispatchPurpose.REPAIR
    )
    return required


def _with_purpose(config: SlotConfig, purpose: DispatchPurpose) -> SlotConfig:
    if config.purpose is purpose:
        return config
    return SlotConfig(
        slot=config.slot,
        provider=config.provider,
        model=config.model,
        endpoint=config.endpoint,
        purpose=purpose,
    )


def build_orchestration_service(
    *,
    settings: Settings,
    gate: PrivacyGate,
    router: ProviderRouter,
    provider_slots: dict[ModelSlot, SlotConfig],
    analyzer: PiiAnalyzer,
) -> OrchestrationService:
    """Assemble the orchestration service from configured slots and settings."""

    slots_by_name = {slot.value: config for slot, config in provider_slots.items()}
    reasoning = tuple(
        slot.value
        for slot in (
            ModelSlot.REASONER_A,
            ModelSlot.REASONER_B,
            ModelSlot.REASONER_C,
        )
        if slot in provider_slots
    )
    # Architecture roles: reasoner_b and reasoner_c are the debate reviewers.
    reviewer = reasoning[1:]
    deps = OrchestrationDependencies(
        router=router,
        reasoning_slots=reasoning,
        reviewer_slots=reviewer,
        adjudicator_slot=ModelSlot.FINAL_ADJUDICATOR.value,
        max_debate_rounds=settings.orchestration_max_debate_rounds,
        max_provider_calls=settings.orchestration_max_provider_calls,
        max_total_tokens=settings.orchestration_max_total_tokens,
        max_output_tokens=settings.llm_max_output_tokens,
        temperature=settings.llm_temperature,
        urgent_rules=tuple(settings.safety_urgent_rules),
        analyzer=analyzer,
    )
    return OrchestrationService(
        gate=gate,
        deps=deps,
        slots=slots_by_name,
        timeout_seconds=settings.orchestration_timeout_seconds,
    )

"""Question-specific minimization and the mandatory fail-closed privacy gate."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from uuid import UUID, uuid4

from app.privacy.digest import destination_sha256, payload_sha256, stable_ref
from app.privacy.engine import PiiAnalysisError, PiiAnalyzer, PiiAnalyzerUnavailable
from app.privacy.policy import (
    QuasiIdentifierReviewer,
    QuestionPolicy,
    classify_question,
    detect_prompt_injection,
    select_conflicts,
    select_findings,
    select_missing_information,
    select_observations,
)
from app.privacy.redaction import TokenVault, redact_text
from app.privacy.store import PrivacyApprovalStore
from app.privacy.temporal import generalize_observed_date
from app.schemas.evidence import EvidencePackage, ParsedObservation
from app.schemas.privacy import (
    ExternalDestination,
    MinimizedConflict,
    MinimizedFinding,
    MinimizedObservation,
    ObservationField,
    PreparedPayload,
    PrivacyAuditRecord,
    PrivacyDecision,
    PrivacyStatus,
    QuestionType,
    utc_now,
)

logger = logging.getLogger(__name__)


class PrivacyGate:
    """Minimize an evidence package and issue destination-bound approvals."""

    def __init__(
        self,
        *,
        analyzer: PiiAnalyzer,
        approvals: PrivacyApprovalStore,
        local_only: bool,
        max_question_chars: int = 2000,
        approval_ttl_seconds: int = 300,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._analyzer = analyzer
        self._approvals = approvals
        self._local_only = local_only
        self._max_question_chars = max_question_chars
        self._approval_ttl_seconds = approval_ttl_seconds
        self._clock = clock
        self._reviewer = QuasiIdentifierReviewer()

    async def prepare(
        self,
        *,
        evidence: EvidencePackage,
        question: str,
        destination: ExternalDestination,
        request_id: str,
        actor_ref: str | None = None,
    ) -> PrivacyDecision:
        """Return approved, rejected, or needs_review; only approved exposes a payload."""

        approval_id = uuid4()
        now = self._clock()
        patient_ref = stable_ref(evidence.patient_id) if evidence.patient_id else None
        actor_hash = stable_ref(actor_ref) if actor_ref else None

        if self._local_only:
            return await self._reject(
                approval_id=approval_id,
                request_id=request_id,
                destination=destination,
                now=now,
                patient_ref=patient_ref,
                actor_ref=actor_hash,
                reason_codes=["local_only_mode"],
            )

        cleaned = question.strip()
        if not cleaned:
            return await self._reject(
                approval_id=approval_id,
                request_id=request_id,
                destination=destination,
                now=now,
                patient_ref=patient_ref,
                actor_ref=actor_hash,
                reason_codes=["empty_question"],
            )
        if len(cleaned) > self._max_question_chars:
            return await self._reject(
                approval_id=approval_id,
                request_id=request_id,
                destination=destination,
                now=now,
                patient_ref=patient_ref,
                actor_ref=actor_hash,
                reason_codes=["question_too_long"],
            )

        provenance_reasons = _provenance_reasons(evidence)
        if provenance_reasons:
            return await self._reject(
                approval_id=approval_id,
                request_id=request_id,
                destination=destination,
                now=now,
                patient_ref=patient_ref,
                actor_ref=actor_hash,
                reason_codes=provenance_reasons,
            )

        try:
            policy = classify_question(cleaned)
            vault = TokenVault()
            redacted = redact_text(
                cleaned,
                analyzer=self._analyzer,
                vault=vault,
                on_date=now.date(),
            )
            observations = select_observations(evidence, policy)
            payload = _build_payload(
                evidence=evidence,
                observations=observations,
                policy=policy,
                request_id=request_id,
                question=redacted.text,
                vault=vault,
            )
            review_flags = self._reviewer.review(
                entity_types=redacted.entity_types,
                redaction_flags=redacted.flags,
                prompt_injection=detect_prompt_injection(cleaned),
                residual_pii=vault.contains_unredacted_value(redacted.text),
            )
        except PiiAnalyzerUnavailable:
            logger.error("privacy analyzer unavailable request_id=%s", request_id)
            return await self._reject(
                approval_id=approval_id,
                request_id=request_id,
                destination=destination,
                now=now,
                patient_ref=patient_ref,
                actor_ref=actor_hash,
                reason_codes=["presidio_unavailable"],
            )
        except PiiAnalysisError:
            logger.error("privacy analysis failed request_id=%s", request_id)
            return await self._reject(
                approval_id=approval_id,
                request_id=request_id,
                destination=destination,
                now=now,
                patient_ref=patient_ref,
                actor_ref=actor_hash,
                reason_codes=["presidio_analysis_failed"],
            )
        except Exception as exc:  # fail closed on every unexpected service path
            logger.error(
                "privacy gate failure request_id=%s type=%s", request_id, type(exc).__name__
            )
            return await self._reject(
                approval_id=approval_id,
                request_id=request_id,
                destination=destination,
                now=now,
                patient_ref=patient_ref,
                actor_ref=actor_hash,
                reason_codes=["gate_failure"],
            )

        if review_flags:
            return await self._finish(
                approval_id=approval_id,
                request_id=request_id,
                destination=destination,
                now=now,
                patient_ref=patient_ref,
                actor_ref=actor_hash,
                status=PrivacyStatus.NEEDS_REVIEW,
                reason_codes=[],
                review_flags=review_flags,
                question_type=policy.question_type,
                observation_count=len(payload.observations),
                entity_counts=redacted.counts,
            )

        return await self._finish(
            approval_id=approval_id,
            request_id=request_id,
            destination=destination,
            now=now,
            patient_ref=patient_ref,
            actor_ref=actor_hash,
            status=PrivacyStatus.APPROVED,
            reason_codes=[],
            question_type=policy.question_type,
            observation_count=len(payload.observations),
            entity_counts=redacted.counts,
            payload=payload,
            expires_at=now + timedelta(seconds=self._approval_ttl_seconds),
        )

    async def _reject(
        self,
        *,
        approval_id: UUID,
        request_id: str,
        destination: ExternalDestination,
        now: datetime,
        patient_ref: str | None,
        actor_ref: str | None,
        reason_codes: Sequence[str],
    ) -> PrivacyDecision:
        """Record one fail-closed rejection without exposing any candidate payload."""

        return await self._finish(
            approval_id=approval_id,
            request_id=request_id,
            destination=destination,
            now=now,
            patient_ref=patient_ref,
            actor_ref=actor_ref,
            status=PrivacyStatus.REJECTED,
            reason_codes=reason_codes,
            question_type=QuestionType.GENERAL,
            observation_count=0,
            entity_counts={},
        )

    async def _finish(
        self,
        *,
        approval_id: UUID,
        request_id: str,
        destination: ExternalDestination,
        now: datetime,
        patient_ref: str | None,
        actor_ref: str | None,
        status: PrivacyStatus,
        reason_codes: Sequence[str],
        question_type: QuestionType,
        observation_count: int,
        entity_counts: dict[str, int],
        review_flags: Sequence[str] = (),
        payload: PreparedPayload | None = None,
        expires_at: datetime | None = None,
    ) -> PrivacyDecision:
        facts = {
            "approval_id": approval_id,
            "request_id": request_id,
            "status": status,
            "purpose": destination.purpose,
            "provider": destination.provider,
            "model": destination.model,
            "destination_digest": destination_sha256(destination),
            "payload_sha256": payload_sha256(payload) if payload is not None else None,
            "patient_ref": patient_ref,
            "actor_ref": actor_ref,
            "question_type": question_type,
            "observation_count": observation_count,
            "reason_codes": list(reason_codes),
            "review_flags": list(review_flags),
            "entity_counts": dict(entity_counts),
            "created_at": now,
            "expires_at": expires_at,
        }
        try:
            record = PrivacyAuditRecord.model_validate(facts)
            await self._approvals.record(record)
        except Exception as exc:
            logger.error(
                "privacy audit storage failed request_id=%s type=%s",
                request_id,
                type(exc).__name__,
            )
            return PrivacyDecision(
                approval_id=approval_id,
                status=PrivacyStatus.REJECTED,
                destination=destination,
                payload=None,
                payload_sha256=None,
                reason_codes=["approval_store_failed"],
            )

        if status is PrivacyStatus.APPROVED and payload is not None:
            decision = PrivacyDecision(
                approval_id=record.approval_id,
                status=status,
                destination=destination,
                payload=payload,
                payload_sha256=record.payload_sha256,
                reason_codes=[],
                review_flags=[],
                expires_at=expires_at,
            )
        else:
            decision = PrivacyDecision(
                approval_id=record.approval_id,
                status=status,
                destination=destination,
                payload=None,
                payload_sha256=None,
                reason_codes=list(reason_codes),
                review_flags=list(review_flags),
                expires_at=None,
            )
        logger.info(
            "privacy decision status=%s request_id=%s question_type=%s observations=%d",
            decision.status.value,
            request_id,
            question_type.value,
            observation_count,
        )
        return decision


def _build_payload(
    *,
    evidence: EvidencePackage,
    observations: Sequence[ParsedObservation],
    policy: QuestionPolicy,
    request_id: str,
    question: str,
    vault: TokenVault,
) -> PreparedPayload:
    allowed_fields = policy.allowed_fields
    question_type = policy.question_type
    minimized: list[MinimizedObservation] = []
    for observation in observations:
        reference_range = None
        if ObservationField.REFERENCE_RANGE in allowed_fields:
            reference_range = _reference_range(observation)
        minimized.append(
            MinimizedObservation(
                ref=vault.token_for("EVIDENCE", observation.evidence_id),
                lab=observation.canonical_name[:128],
                value=(
                    observation.value_lexeme[:64]
                    if ObservationField.VALUE in allowed_fields
                    and observation.value_lexeme is not None
                    else None
                ),
                unit=(
                    observation.unit_lexeme[:64]
                    if ObservationField.UNIT in allowed_fields
                    and observation.unit_lexeme is not None
                    else None
                ),
                reference_range=reference_range,
                date=(
                    generalize_observed_date(observation.observed_date)
                    if ObservationField.DATE in allowed_fields
                    else None
                ),
                status=observation.status.value,
                flags=(
                    [flag.value for flag in observation.flags]
                    if ObservationField.FLAGS in allowed_fields
                    else []
                ),
                outside_reference_range=(
                    observation.outside_reference_range
                    if ObservationField.REFERENCE_RANGE in allowed_fields
                    else None
                ),
            )
        )

    findings = [
        MinimizedFinding(
            lab=finding.canonical_name[:128],
            status=finding.status.value,
            refs=[
                vault.token_for("EVIDENCE", evidence_id)
                for evidence_id in finding.evidence_ids
            ],
        )
        for finding in select_findings(evidence, observations=observations, policy=policy)
    ]
    conflicts = [
        MinimizedConflict(
            lab=conflict.canonical_name[:128],
            values=(
                [value[:64] for value in conflict.value_lexemes]
                if ObservationField.VALUE in allowed_fields
                else []
            ),
            units=(
                [unit[:64] for unit in conflict.unit_lexemes]
                if ObservationField.UNIT in allowed_fields
                else []
            ),
            refs=[
                vault.token_for("EVIDENCE", evidence_id)
                for evidence_id in conflict.evidence_ids
            ],
        )
        for conflict in select_conflicts(evidence, observations=observations, policy=policy)
    ]
    return PreparedPayload(
        request_id=request_id,
        question=question,
        question_type=question_type,
        observations=minimized,
        findings=findings,
        conflicts=conflicts,
        missing_information=select_missing_information(
            evidence, observations=observations, policy=policy
        )[:64],
    )


def _reference_range(observation: ParsedObservation) -> str | None:
    if (
        observation.reference_low_lexeme is not None
        and observation.reference_high_lexeme is not None
    ):
        value = f"{observation.reference_low_lexeme}-{observation.reference_high_lexeme}"
        if observation.reference_unit_lexeme is not None:
            value = f"{value} {observation.reference_unit_lexeme}"
        return value[:128]
    if observation.reference_range_lexeme is not None:
        return observation.reference_range_lexeme[:128]
    return None


def _provenance_reasons(evidence: EvidencePackage) -> list[str]:
    reasons: list[str] = []
    source_keys = {
        (source.document_id, source.chunk_id, source.content_hash, source.page_number)
        for source in evidence.sources
    }
    known_ids: set[str] = set()
    for observation in evidence.observations:
        source = evidence.resolve(observation.evidence_id)
        if source is None:
            reasons.append("unknown_evidence_reference")
            continue
        known_ids.add(observation.evidence_id)
        if (
            source.document_id,
            source.chunk_id,
            source.content_hash,
            source.page_number,
        ) not in source_keys:
            reasons.append("unresolved_provenance")
        if observation.value_lexeme is not None:
            try:
                parsed = Decimal(observation.value_lexeme)
            except InvalidOperation:
                reasons.append("invalid_value_lexeme")
            else:
                if observation.value_decimal != parsed:
                    reasons.append("value_mismatch")
        if (
            evidence.as_of is not None
            and observation.observed_date is not None
            and observation.observed_date > evidence.as_of.date()
        ):
            reasons.append("observation_after_as_of")
    for finding in evidence.findings:
        if any(evidence_id not in known_ids for evidence_id in finding.evidence_ids):
            reasons.append("unknown_finding_reference")
    for conflict in evidence.conflicts:
        if any(evidence_id not in known_ids for evidence_id in conflict.evidence_ids):
            reasons.append("unknown_conflict_reference")
    return list(dict.fromkeys(reasons))

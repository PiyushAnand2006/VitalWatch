"""Standalone verifier tests: each deterministic check in isolation."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from app.orchestration.context import (
    build_verification_context,
    candidate_contradictions,
    verify_claim,
)
from app.orchestration.safety_rules import evaluate_urgent_rules
from app.orchestration.verifiers.consistency import (
    find_internal_contradictions,
    verify_consistency,
)
from app.orchestration.verifiers.evidence import verify_evidence
from app.orchestration.verifiers.numeric import verify_numeric
from app.orchestration.verifiers.provenance import verify_provenance
from app.orchestration.verifiers.safety import verify_safety
from app.orchestration.verifiers.schema import (
    SchemaError,
    extract_json_object,
    parse_adjudication,
    parse_candidate,
)
from app.orchestration.verifiers.temporal import verify_temporal
from app.schemas.evidence import EvidencePackage, LabConflict
from app.schemas.orchestration import (
    CandidateClaim,
    CandidateConflict,
    ReasoningCandidate,
    UrgentRule,
)
from app.schemas.privacy import (
    MinimizedConflict,
    MinimizedObservation,
    PreparedPayload,
    QuestionType,
)
from tests.privacy_fixtures import make_observation, make_source

DOC_ID = make_source().document_id


def make_payload(*, conflict: bool = False) -> PreparedPayload:
    observations = [
        MinimizedObservation(
            ref="[EV_1]",
            lab="Potassium",
            value="5.4",
            unit="mmol/L",
            reference_range="3.5-5.1 mmol/L",
            date="2026-01",
            status="explicit",
            flags=["outside_reference_range"],
            outside_reference_range=True,
        )
    ]
    conflicts = []
    if conflict:
        observations.append(
            MinimizedObservation(
                ref="[EV_2]",
                lab="Potassium",
                value="6.1",
                unit="mmol/L",
                reference_range="3.5-5.1 mmol/L",
                date="2026-01",
                status="explicit",
            )
        )
        conflicts = [
            MinimizedConflict(
                lab="Potassium",
                values=["5.4", "6.1"],
                units=["mmol/L"],
                refs=["[EV_1]", "[EV_2]"],
            )
        ]
    return PreparedPayload(
        request_id="req-verify",
        question="Interpret the potassium result.",
        question_type=QuestionType.LAB_VALUE,
        observations=observations,
        conflicts=conflicts,
    )


def make_evidence_for(payload: PreparedPayload) -> EvidencePackage:
    observations = []
    for observation in payload.observations:
        value = observation.value
        observations.append(
            make_observation(
                canonical_name=observation.lab,
                value=value,
                unit=observation.unit,
                low="3.5",
                high="5.1",
                observed_date=date(2026, 1, 5),
            )
        )
    if payload.conflicts:
        evidence_ids = [observation.evidence_id for observation in observations]
        conflicts = [
            LabConflict(
                canonical_name="Potassium",
                value_lexemes=["5.4", "6.1"],
                unit_lexemes=["mmol/L"],
                evidence_ids=evidence_ids,
            )
        ]
    else:
        conflicts = []
    return EvidencePackage(
        query="synthetic query",
        patient_id="synthetic-patient",
        as_of=datetime(2026, 11, 10, tzinfo=UTC),
        retrieval_status="retrieved",
        filters={"as_of": datetime(2026, 11, 10, tzinfo=UTC)},
        observations=observations,
        findings=[],
        conflicts=conflicts,
        missing_information=[],
        excluded_after_as_of=[],
        sources=list({observation.source for observation in observations}),
    )


@pytest.fixture
def context():
    payload = make_payload()
    evidence = make_evidence_for(payload)
    return build_verification_context(payload, evidence)

def claim(text: str, refs: tuple[str, ...] = ("[EV_1]",)) -> CandidateClaim:
    return CandidateClaim(text=text, evidence_refs=refs)


# --------------------------------------------------------------------------
# Schema verifier
# --------------------------------------------------------------------------


def test_schema_parses_a_valid_candidate() -> None:
    parsed = parse_candidate(
        '{"facts": [{"text": "Potassium 5.4 mmol/L.", "evidence_refs": ["[EV_1]"]}],'
        ' "hypotheses": [], "conflicts": []}'
    )
    assert parsed.facts[0].evidence_refs == ("[EV_1]",)


def test_schema_accepts_fenced_json() -> None:
    fenced = '```json\n{"facts": [], "hypotheses": [], "conflicts": []}\n```'
    assert parse_candidate(fenced).facts == []


def test_schema_rejects_non_json_and_non_objects() -> None:
    with pytest.raises(SchemaError):
        parse_candidate("not json at all")
    with pytest.raises(SchemaError):
        extract_json_object('["a", "list"]')


def test_schema_rejects_extra_fields_and_bad_refs() -> None:
    with pytest.raises(SchemaError):
        parse_candidate('{"facts": [], "surprise_field": true}')
    with pytest.raises(SchemaError):
        parse_candidate(
            '{"facts": [{"text": "x", "evidence_refs": ["EV1"]}]}'
        )  # not token-shaped
    with pytest.raises(SchemaError):
        parse_candidate(
            '{"facts": [{"text": "x", "evidence_refs": []}]}'
        )  # citations required


def test_schema_parses_adjudication_strictly() -> None:
    parsed = parse_adjudication(
        '{"verified_facts": [{"text": "t", "evidence_refs": ["[EV_1]"]}],'
        ' "unresolved_disagreements": ["lab differs"]}'
    )
    assert parsed.unresolved_disagreements == ["lab differs"]


# --------------------------------------------------------------------------
# Evidence verifier
# --------------------------------------------------------------------------


def test_evidence_rejects_unknown_reference(context) -> None:
    result = verify_evidence(
        claim_text="Potassium was 5.4 mmol/L.",
        evidence_refs=("[EV_9]",),
        known_refs=context.known_refs,
        ref_labs=context.ref_labs,
    )
    assert result.passed is False
    assert result.reason_codes == ["unknown_reference"]


def test_evidence_rejects_unsupported_claims(context) -> None:
    result = verify_evidence(
        claim_text="The patient looks fine.",
        evidence_refs=("[EV_1]",),
        known_refs=context.known_refs,
        ref_labs=context.ref_labs,
    )
    assert result.passed is False
    assert result.reason_codes == ["unsupported_reference"]


def test_evidence_accepts_lab_supported_claim(context) -> None:
    result = verify_evidence(
        claim_text="Potassium was above range.",
        evidence_refs=("[EV_1]",),
        known_refs=context.known_refs,
        ref_labs=context.ref_labs,
    )
    assert result.passed is True


# --------------------------------------------------------------------------
# Numeric verifier
# --------------------------------------------------------------------------


def test_numeric_accepts_exact_value_and_unit() -> None:
    observation = make_payload().observations[0]
    result = verify_numeric(
        claim_text="Potassium was 5.4 mmol/L.", cited_observations=[observation]
    )
    assert result.passed is True
    assert result.matched_values == ["5.4 mmol/L"]


def test_numeric_accepts_reference_bounds() -> None:
    observation = make_payload().observations[0]
    result = verify_numeric(
        claim_text="The range was 3.5-5.1 mmol/L.", cited_observations=[observation]
    )
    assert result.passed is True


def test_numeric_rejects_wrong_decimal() -> None:
    observation = make_payload().observations[0]
    result = verify_numeric(
        claim_text="Potassium was 5.39 mmol/L.", cited_observations=[observation]
    )
    assert result.passed is False
    assert result.reason_codes == ["value_mismatch"]


def test_numeric_rejects_wrong_unit() -> None:
    observation = make_payload().observations[0]
    result = verify_numeric(
        claim_text="Potassium was 5.4 mg/dL.", cited_observations=[observation]
    )
    assert result.passed is False


def test_numeric_ignores_bare_integer_counts() -> None:
    observation = make_payload().observations[0]
    result = verify_numeric(
        claim_text="1 abnormal result was noted.", cited_observations=[observation]
    )
    assert result.passed is True


def test_numeric_rejects_invented_standalone_decimal() -> None:
    observation = make_payload().observations[0]
    result = verify_numeric(
        claim_text="Potassium rose 2.5 from baseline.", cited_observations=[observation]
    )
    assert result.passed is False


# --------------------------------------------------------------------------
# Temporal verifier
# --------------------------------------------------------------------------


def test_temporal_accepts_cited_month() -> None:
    observation = make_payload().observations[0]
    result = verify_temporal(
        claim_text="The 2026-01 report lists it.",
        cited_observations=[observation],
        as_of_year_month=(2026, 11),
    )
    assert result.passed is True


def test_temporal_accepts_month_names() -> None:
    observation = make_payload().observations[0]
    result = verify_temporal(
        claim_text="The January 2026 report lists it.",
        cited_observations=[observation],
        as_of_year_month=(2026, 11),
    )
    assert result.passed is True


def test_temporal_rejects_invented_dates() -> None:
    observation = make_payload().observations[0]
    result = verify_temporal(
        claim_text="The 2025-08 report lists it.",
        cited_observations=[observation],
        as_of_year_month=(2026, 11),
    )
    assert result.passed is False
    assert result.reason_codes == ["unsupported_date"]


def test_temporal_rejects_dates_after_as_of() -> None:
    observation = make_payload().observations[0]
    observation = observation.model_copy(update={"date": "2027-01"})
    result = verify_temporal(
        claim_text="The January 2027 report lists it.",
        cited_observations=[observation],
        as_of_year_month=(2026, 11),
    )
    assert result.passed is False
    assert result.reason_codes == ["date_after_as_of"]


# --------------------------------------------------------------------------
# Consistency verifier
# --------------------------------------------------------------------------


def test_consistency_requires_conflict_surfacing(context) -> None:
    payload = make_payload(conflict=True)
    evidence = make_evidence_for(payload)
    conflict_context = build_verification_context(payload, evidence)
    candidate = ReasoningCandidate(conflicts=[])
    result = verify_consistency(
        claim=claim("Potassium was 5.4 mmol/L."),
        candidate=candidate,
        cited_observations=payload.observations[:1],
        payload_conflicts=payload.conflicts,
    )
    assert result.passed is False
    assert result.reason_codes == ["conflict_not_surfaced"]

    surfacing = ReasoningCandidate(
        facts=[claim("Potassium was 5.4 mmol/L.")],
        conflicts=[
            CandidateConflict(
                lab="Potassium",
                description="Source reports differ.",
                evidence_refs=("[EV_1]", "[EV_2]"),
            )
        ],
    )
    ok = verify_consistency(
        claim=claim("Potassium was 5.4 mmol/L."),
        candidate=surfacing,
        cited_observations=payload.observations[:1],
        payload_conflicts=payload.conflicts,
    )
    assert ok.passed is True
    assert conflict_context.known_refs == {"[EV_1]", "[EV_2]"}


def test_internal_contradiction_finder_groups_by_lab() -> None:
    ref_labs = {"[EV_1]": frozenset({"Potassium"}), "[EV_2]": frozenset({"Sodium"})}
    ref_values = {
        "[EV_1]": frozenset({("5.4", "mmol/L"), ("6.1", "mmol/L")}),
        "[EV_2]": frozenset({("140", "mmol/L")}),
    }
    candidate = ReasoningCandidate(
        facts=[
            claim("Potassium was 5.4 mmol/L.", ("[EV_1]",)),
            claim("Potassium was 6.1 mmol/L.", ("[EV_1]",)),
        ]
    )
    contradictions = find_internal_contradictions(candidate, ref_labs, ref_values)
    assert contradictions == ["Potassium: 5.4 mmol/L, 6.1 mmol/L"]

    # Reference-range bounds mentioned alongside the value are not contradictions.
    with_bounds = ReasoningCandidate(
        facts=[
            claim(
                "Potassium was 5.4 mmol/L, above the 3.5-5.1 mmol/L range.",
                ("[EV_1]",),
            )
        ]
    )
    assert find_internal_contradictions(with_bounds, ref_labs, ref_values) == []

    different_labs = ReasoningCandidate(
        facts=[
            claim("Potassium was 5.4 mmol/L.", ("[EV_1]",)),
            claim("Sodium was 140 mmol/L.", ("[EV_2]",)),
        ]
    )
    assert find_internal_contradictions(different_labs, ref_labs, ref_values) == []


def test_candidate_contradictions_use_the_context(context) -> None:
    agreeing = ReasoningCandidate(
        facts=[claim("Potassium was 5.4 mmol/L."), claim("Potassium was 5.4 mmol/L.")]
    )
    assert candidate_contradictions(agreeing, context) == []

    # An invented value is a numeric failure, not a contradiction, and is not
    # counted by the contradiction finder.
    invented = ReasoningCandidate(
        facts=[claim("Potassium was 5.4 mmol/L."), claim("Potassium was 5.39 mmol/L.")]
    )
    assert candidate_contradictions(invented, context) == []

    payload = make_payload(conflict=True)
    conflict_context = build_verification_context(payload, make_evidence_for(payload))
    competing = ReasoningCandidate(
        facts=[
            claim("Potassium was 5.4 mmol/L.", ("[EV_1]",)),
            claim("Potassium was 6.1 mmol/L.", ("[EV_2]",)),
        ]
    )
    assert candidate_contradictions(competing, conflict_context) == [
        "Potassium: 5.4 mmol/L, 6.1 mmol/L"
    ]


# --------------------------------------------------------------------------
# Safety verifier
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("You must take insulin daily.", "treatment_advice"),
        ("Start medication immediately.", "treatment_advice"),
        ("I prescribe metformin for this.", "treatment_advice"),
        ("This definitely indicates a disease.", "diagnostic_certainty"),
        ("This confirms the diagnosis of deficiency.", "diagnostic_certainty"),
    ],
)
def test_safety_rejects_unsafe_text(text: str, reason: str) -> None:
    result = verify_safety(claim_text=text)
    assert result.passed is False
    assert reason in result.reason_codes


@pytest.mark.parametrize(
    "text",
    [
        "Potassium was 5.4 mmol/L, above the reference range.",
        "The result may reflect sampling variation.",
        "Clinician review is required.",
        "The patient should be reviewed by a clinician.",
    ],
)
def test_safety_accepts_cautious_text(text: str) -> None:
    assert verify_safety(claim_text=text).passed is True


# --------------------------------------------------------------------------
# Provenance verifier
# --------------------------------------------------------------------------


def test_provenance_resolves_through_the_context(context) -> None:
    result = verify_provenance(
        evidence_refs=("[EV_1]",),
        token_map=context.token_map,
        evidence=context.evidence,
        known_source_keys=context.known_source_keys,
    )
    assert result.passed is True


def test_provenance_rejects_unmatched_tokens(context) -> None:
    result = verify_provenance(
        evidence_refs=("[EV_7]",),
        token_map=context.token_map,
        evidence=context.evidence,
        known_source_keys=context.known_source_keys,
    )
    assert result.passed is False
    assert result.reason_codes == ["unresolved_reference"]


# --------------------------------------------------------------------------
# Full claim verification pipeline
# --------------------------------------------------------------------------


def test_verify_claim_passes_a_fully_supported_fact(context) -> None:
    verdict = verify_claim(
        claim=claim("Potassium was 5.4 mmol/L, above the 3.5-5.1 mmol/L range."),
        candidate=ReasoningCandidate(facts=[claim("x")]),
        context=context,
        is_fact=True,
    )
    assert verdict.passed is True
    assert [result.verifier.value for result in verdict.results] == [
        "schema",
        "evidence",
        "numeric",
        "temporal",
        "consistency",
        "safety",
        "provenance",
    ]


def test_verify_claim_fails_on_wrong_value(context) -> None:
    verdict = verify_claim(
        claim=claim("Potassium was 5.9 mmol/L."),
        candidate=ReasoningCandidate(facts=[claim("x")]),
        context=context,
        is_fact=True,
    )
    failed = {result.verifier.value for result in verdict.results if not result.passed}
    assert failed == {"numeric"}


# --------------------------------------------------------------------------
# Urgent safety rules
# --------------------------------------------------------------------------


def test_urgent_rule_triggers_on_configured_threshold() -> None:
    rules = (
        UrgentRule(
            label="severe_hyperkalemia",
            lab_pattern="potassium",
            operator="gt",
            value="7.0",
            unit="mmol/L",
        ),
    )
    payload = make_payload().model_copy(
        update={
            "observations": [
                make_payload().observations[0].model_copy(update={"value": "7.5"})
            ]
        }
    )
    findings = evaluate_urgent_rules(payload, rules)
    assert [finding.label for finding in findings] == ["severe_hyperkalemia"]
    assert findings[0].observation_ref == "[EV_1]"


def test_urgent_rule_ignores_other_units_and_labs() -> None:
    rules = (
        UrgentRule(
            label="severe_hyperkalemia",
            lab_pattern="potassium",
            operator="gt",
            value="7.0",
            unit="mg/dL",
        ),
    )
    assert evaluate_urgent_rules(make_payload(), rules) == []


def test_urgent_rule_operators() -> None:
    payload = make_payload()
    low = UrgentRule(label="low", lab_pattern="potassium", operator="lt", value="3.5")
    high = UrgentRule(label="high", lab_pattern="potassium", operator="gte", value="5.1")
    assert evaluate_urgent_rules(payload, (low,)) == []
    assert [
        finding.label for finding in evaluate_urgent_rules(payload, (high,))
    ] == ["high"]


def test_invalid_urgent_rule_regex_is_rejected_at_load() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        UrgentRule(label="bad", lab_pattern="([unclosed", operator="gt", value="1")

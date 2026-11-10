"""Synthetic tests for exact, conservative lab parsing and evidence packages."""

from datetime import UTC, date, datetime
from decimal import Decimal
from hashlib import sha256
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.schemas.embeddings import EmbeddingModelInfo
from app.schemas.evidence import (
    EvidenceFlag,
    EvidencePackage,
    EvidenceSource,
    FindingStatus,
    ObservationStatus,
    ParsedObservation,
)
from app.schemas.retrieval import (
    NoEvidenceReason,
    RetrievalFilter,
    RetrievalResult,
    RetrievalStatus,
    RetrievedChunk,
)
from app.services.evidence import EvidenceCompiler, parse_lab_observations

MODEL = EmbeddingModelInfo(
    name="sentence-transformers/all-MiniLM-L6-v2", revision="synthetic-revision", dimension=384
)
DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")
CREATED_AT = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)


def _source(
    *, chunk_id: str = "p1-c1", quality: str = "high", document_id: UUID = DOCUMENT_ID
) -> EvidenceSource:
    return EvidenceSource(
        document_id=document_id,
        chunk_id=chunk_id,
        page_number=1,
        content_hash=sha256(chunk_id.encode()).hexdigest(),
        method="text",
        quality=quality,
        document_created_at=CREATED_AT,
        score=0.9,
    )


def _observation(text: str, **kwargs: object) -> ParsedObservation:
    observations = parse_lab_observations(text, source=_source(**kwargs))  # type: ignore[arg-type]
    assert observations, text
    return observations[0]


def _retrieval(*chunks: tuple[str, str], quality: str = "high") -> RetrievalResult:
    items = [
        RetrievedChunk(
            chunk_id=chunk_id,
            document_id=DOCUMENT_ID,
            patient_id="synthetic-patient-a",
            page_number=1,
            section=None,
            text=text,
            content_hash=sha256(text.encode()).hexdigest(),
            method="text",
            quality=quality,
            score=0.9,
            distance=0.1,
            document_created_at=CREATED_AT,
        )
        for chunk_id, text in chunks
    ]
    return RetrievalResult(
        query="synthetic query",
        patient_id="synthetic-patient-a",
        status=RetrievalStatus.RETRIEVED,
        items=items,
        candidates_considered=len(items),
        duplicates_removed=0,
        below_threshold=0,
        top_k=5,
        min_score=0.25,
        model=MODEL,
        filters=RetrievalFilter(),
    )


def _empty_retrieval() -> RetrievalResult:
    return RetrievalResult(
        query="synthetic query",
        patient_id="synthetic-patient-a",
        status=RetrievalStatus.NO_EVIDENCE,
        reason=NoEvidenceReason.NO_MATCHING_DOCUMENTS,
        items=[],
        candidates_considered=0,
        duplicates_removed=0,
        below_threshold=0,
        top_k=5,
        min_score=0.25,
        model=MODEL,
        filters=RetrievalFilter(),
    )


def test_exact_decimal_signs_and_units_are_preserved() -> None:
    observations = parse_lab_observations(
        "Sodium: 140 mmol/L\nPotassium 4.10 mmol/L\nGlucose: -0.50 mmol/L\n",
        source=_source(),
    )

    assert [observation.value_lexeme for observation in observations] == ["140", "4.10", "-0.50"]
    assert [observation.unit_lexeme for observation in observations] == [
        "mmol/L",
        "mmol/L",
        "mmol/L",
    ]
    assert observations[1].value_decimal == Decimal("4.10")
    assert observations[2].value_decimal == Decimal("-0.50")
    assert all(observation.status is ObservationStatus.EXPLICIT for observation in observations)


def test_reference_range_is_compared_only_with_an_explicit_matching_unit() -> None:
    high = _observation("Sodium 150 mmol/L (ref 135-145 mmol/L)")
    normal = _observation("Sodium 140 mmol/L (ref 135-145 mmol/L)")

    assert high.outside_reference_range is True
    assert EvidenceFlag.OUTSIDE_REFERENCE_RANGE in high.flags
    assert normal.outside_reference_range is False
    assert EvidenceFlag.OUTSIDE_REFERENCE_RANGE not in normal.flags


def test_reference_range_without_a_unit_is_not_interpreted() -> None:
    observation = _observation("Sodium 150 mmol/L (135-145)")

    assert observation.outside_reference_range is None
    assert EvidenceFlag.RANGE_UNIT_MISSING in observation.flags


def test_reference_range_with_a_different_unit_is_not_interpreted() -> None:
    observation = _observation("Creatinine 1.2 mg/dL (ref 60-110 umol/L)")

    assert observation.outside_reference_range is None
    assert EvidenceFlag.RANGE_UNIT_MISMATCH in observation.flags


def test_missing_unit_is_uncertain_and_never_inferred() -> None:
    observation = _observation("Creatinine 1.2")

    assert observation.unit_lexeme is None
    assert observation.status is ObservationStatus.UNCERTAIN
    assert EvidenceFlag.MISSING_UNIT in observation.flags


def test_uncertain_ocr_text_is_marked_uncertain() -> None:
    observation = _observation("Sodium 140 mmol/L", quality="uncertain")

    assert observation.status is ObservationStatus.UNCERTAIN
    assert EvidenceFlag.OCR_UNCERTAIN in observation.flags


def test_pending_lab_is_missing_information() -> None:
    observation = _observation("Potassium: pending")

    assert observation.value_lexeme is None
    assert observation.status is ObservationStatus.MISSING
    package = EvidenceCompiler().compile(
        retrieval=_retrieval(("p1-c1", "Potassium: pending"))
    )
    assert package.missing_information == ["Potassium"]


def test_conflicting_values_are_preserved_not_resolved() -> None:
    package = EvidenceCompiler().compile(
        retrieval=_retrieval(
            ("p1-c1", "Sodium 140 mmol/L"),
            ("p1-c2", "Sodium 148 mmol/L"),
        )
    )

    assert len(package.conflicts) == 1
    conflict = package.conflicts[0]
    assert conflict.canonical_name == "Sodium"
    assert conflict.value_lexemes == ["140", "148"]
    finding = next(item for item in package.findings if item.canonical_name == "Sodium")
    assert finding.status is FindingStatus.CONFLICTING
    assert len(finding.evidence_ids) == 2


def test_conflicting_units_are_preserved_not_converted() -> None:
    package = EvidenceCompiler().compile(
        retrieval=_retrieval(
            ("p1-c1", "Creatinine 1.2 mg/dL"),
            ("p1-c2", "Creatinine 1.2 umol/L"),
        )
    )

    assert len(package.conflicts) == 1
    assert package.conflicts[0].unit_lexemes == ["mg/dL", "umol/L"]


def test_identical_values_are_not_reported_as_conflicts() -> None:
    package = EvidenceCompiler().compile(
        retrieval=_retrieval(
            ("p1-c1", "Sodium 140 mmol/L"),
            ("p1-c2", "Sodium 140 mmol/L"),
        )
    )

    assert package.conflicts == []
    finding = next(item for item in package.findings if item.canonical_name == "Sodium")
    assert finding.status is FindingStatus.EXPLICIT


def test_iso_and_slash_dates_are_preserved_exactly() -> None:
    iso = _observation("Sodium 140 mmol/L 2026-01-05")
    slash = _observation("Sodium 140 mmol/L 2026/01/05")

    assert iso.observed_date_lexeme == "2026-01-05"
    assert iso.observed_date == date(2026, 1, 5)
    assert slash.observed_date == date(2026, 1, 5)


def test_multiple_dates_in_one_chunk_are_ambiguous() -> None:
    observation = _observation("Sodium 140 mmol/L 2026-01-05 and 2026-01-06")

    assert observation.observed_date is None
    assert EvidenceFlag.DATE_AMBIGUOUS in observation.flags


def test_unparseable_date_is_flagged() -> None:
    observation = _observation("Sodium 140 mmol/L 2026-02-31")

    assert observation.observed_date is None
    assert EvidenceFlag.DATE_UNPARSEABLE in observation.flags


def test_ambiguous_slash_dates_are_not_guessed() -> None:
    observation = _observation("Sodium 140 mmol/L 05/01/2026")

    assert observation.observed_date is None
    assert observation.observed_date_lexeme is None


def test_evidence_ids_are_stable_and_provenance_bound() -> None:
    first = parse_lab_observations("Sodium 140 mmol/L", source=_source())
    second = parse_lab_observations("Sodium 140 mmol/L", source=_source())
    other_chunk = parse_lab_observations("Sodium 140 mmol/L", source=_source(chunk_id="p1-c2"))

    assert first[0].evidence_id == second[0].evidence_id
    assert first[0].evidence_id != other_chunk[0].evidence_id


def test_observations_after_as_of_are_excluded() -> None:
    package = EvidenceCompiler().compile(
        retrieval=_retrieval(("p1-c1", "Sodium 140 mmol/L 2026-01-05")),
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert package.observations == []
    assert len(package.excluded_after_as_of) == 1
    assert package.excluded_after_as_of[0].observed_date == date(2026, 1, 5)


def test_observation_on_as_of_date_is_kept() -> None:
    package = EvidenceCompiler().compile(
        retrieval=_retrieval(("p1-c1", "Sodium 140 mmol/L 2026-01-05")),
        as_of=datetime(2026, 1, 5, 23, 0, tzinfo=UTC),
    )

    assert len(package.observations) == 1


def test_package_resolves_evidence_to_source_provenance() -> None:
    package = EvidenceCompiler().compile(
        retrieval=_retrieval(("p1-c1", "Sodium 140 mmol/L"))
    )
    evidence_id = package.observations[0].evidence_id

    source = package.resolve(evidence_id)

    assert source is not None
    assert source.document_id == DOCUMENT_ID
    assert source.chunk_id == "p1-c1"
    assert "untrusted data" in package.untrusted_text_notice


def test_no_evidence_package_is_typed_and_empty() -> None:
    package = EvidenceCompiler().compile(retrieval=_empty_retrieval())

    assert package.retrieval_status is RetrievalStatus.NO_EVIDENCE
    assert package.observations == []
    assert package.findings == []
    assert package.sources == []


def test_prose_and_unrelated_lines_are_ignored() -> None:
    observations = parse_lab_observations(
        "Glucose levels were normal.\nKnown allergies: none\n"
        "Patient reports feeling better.\nSodium 140 mmol/L",
        source=_source(),
    )

    assert [observation.canonical_name for observation in observations] == ["Sodium"]


def test_parenthetical_units_and_aliases_are_supported() -> None:
    observations = parse_lab_observations(
        "Na+ 138 mmol/L\nWhite blood cell count (WBC) 5.2 x10^9/L", source=_source()
    )

    assert [observation.canonical_name for observation in observations] == [
        "Sodium",
        "White blood cell count",
    ]
    assert observations[1].unit_lexeme == "x10^9/L"


def test_malformed_contracts_are_rejected() -> None:
    observation = _observation("Sodium 140 mmol/L")
    with pytest.raises(ValidationError):
        ParsedObservation.model_validate(
            {**observation.model_dump(), "evidence_id": "not-an-evidence-id"}
        )
    with pytest.raises(ValidationError):
        ParsedObservation.model_validate({**observation.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError):
        RetrievalFilter(as_of=datetime(2026, 1, 1))
    with pytest.raises(ValidationError):
        RetrievalResult(
            query="synthetic query",
            patient_id="synthetic-patient-a",
            status=RetrievalStatus.NO_EVIDENCE,
            reason=NoEvidenceReason.NO_MATCHING_DOCUMENTS,
            items=_retrieval(("p1-c1", "Sodium 140 mmol/L")).items,
            candidates_considered=1,
            duplicates_removed=0,
            below_threshold=0,
            top_k=5,
            min_score=0.25,
            model=MODEL,
            filters=RetrievalFilter(),
        )


def test_evidence_package_is_immutable() -> None:
    package = EvidenceCompiler().compile(
        retrieval=_retrieval(("p1-c1", "Sodium 140 mmol/L"))
    )

    with pytest.raises(ValidationError):
        package.query = "mutated"  # type: ignore[misc]


def test_unknown_evidence_id_does_not_resolve() -> None:
    package: EvidencePackage = EvidenceCompiler().compile(
        retrieval=_retrieval(("p1-c1", "Sodium 140 mmol/L"))
    )

    assert package.resolve("ev_" + "0" * 24) is None

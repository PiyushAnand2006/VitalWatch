"""Deterministic local parsing of untrusted retrieved text into evidence packages."""

import re
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256

from app.schemas.evidence import (
    EvidenceFlag,
    EvidencePackage,
    EvidenceSource,
    ExcludedObservation,
    FindingStatus,
    LabConflict,
    LabFinding,
    ObservationStatus,
    ParsedObservation,
)
from app.schemas.retrieval import RetrievalResult

_CANONICAL_LABS: dict[str, str] = {
    "sodium": "Sodium",
    "na": "Sodium",
    "na+": "Sodium",
    "potassium": "Potassium",
    "k": "Potassium",
    "k+": "Potassium",
    "chloride": "Chloride",
    "cl": "Chloride",
    "cl-": "Chloride",
    "bicarbonate": "Bicarbonate",
    "hco3": "Bicarbonate",
    "co2": "Bicarbonate",
    "glucose": "Glucose",
    "blood glucose": "Glucose",
    "creatinine": "Creatinine",
    "urea": "Urea",
    "bun": "Urea",
    "blood urea nitrogen": "Urea",
    "hemoglobin": "Hemoglobin",
    "haemoglobin": "Hemoglobin",
    "hb": "Hemoglobin",
    "hgb": "Hemoglobin",
    "hematocrit": "Hematocrit",
    "hct": "Hematocrit",
    "platelets": "Platelets",
    "platelet count": "Platelets",
    "plt": "Platelets",
    "white blood cell count": "White blood cell count",
    "wbc": "White blood cell count",
    "leukocytes": "White blood cell count",
    "lactate": "Lactate",
    "lactic acid": "Lactate",
    "crp": "C-reactive protein",
    "c-reactive protein": "C-reactive protein",
    "inr": "INR",
    "calcium": "Calcium",
    "magnesium": "Magnesium",
    "phosphate": "Phosphate",
    "albumin": "Albumin",
    "bilirubin": "Bilirubin",
    "total bilirubin": "Bilirubin",
    "ast": "AST",
    "alt": "ALT",
    "alkaline phosphatase": "Alkaline phosphatase",
    "alp": "Alkaline phosphatase",
    "troponin": "Troponin",
    "troponin i": "Troponin",
    "troponin t": "Troponin",
    "tsh": "TSH",
    "hba1c": "HbA1c",
}
_ALIASES = sorted(_CANONICAL_LABS, key=len, reverse=True)

_NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
_UNIT = (
    r"(?:[×x]\s*10\s*\^?\s*\d+\s*/\s*[A-Za-z]+"
    r"|10\s*\^?\s*\d+\s*/\s*[A-Za-z]+"
    r"|[A-Za-zµμ%][A-Za-z0-9µμ%^/*.\-]{0,19})"
)
_UNIT_STOPWORDS = {"to", "and", "ref", "reference", "range", "normal", "high", "low"}

_VALUE_RE = re.compile(rf"^\s*[:=]?\s*(?P<value>{_NUMBER})(?![\d.])")
_UNIT_RE = re.compile(rf"^\s*(?P<unit>{_UNIT})")
_PARENTHESIZED_RANGE_RE = re.compile(
    rf"\(\s*(?:ref(?:erence)?\s*[:.]?\s*)?(?P<low>{_NUMBER})\s*(?:-|–|—|\bto\b)\s*"
    rf"(?P<high>{_NUMBER})(?:\s*(?P<unit>{_UNIT}))?\s*\)",
    re.IGNORECASE,
)
_REFERENCE_RANGE_RE = re.compile(
    rf"\bref(?:erence)?(?:\s+range)?\s*[:.]?\s*(?P<low>{_NUMBER})\s*"
    rf"(?:-|–|—|\bto\b)\s*(?P<high>{_NUMBER})(?:\s*(?P<unit>{_UNIT}))?",
    re.IGNORECASE,
)
_PENDING_RE = re.compile(
    r"^(?:pending|not\s+(?:measured|done|available|reported)|awaiting(?:\s+results?)?|n/?a)$",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^[\-*\u2022]+\s*")
_PARENTHETICAL_RE = re.compile(r"^\([^)]{0,24}\)\s*[:=]?\s*")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_SLASH_DATE_RE = re.compile(r"\b(\d{4})/(\d{2})/(\d{2})\b")


def parse_lab_observations(text: str, *, source: EvidenceSource) -> list[ParsedObservation]:
    """Parse common lab lines conservatively, preserving exact source lexemes."""

    date_lexeme, observed_date, date_flags = _chunk_date(text)
    observations: list[ParsedObservation] = []
    seen: set[str] = set()
    for line in text.splitlines():
        matched = _match_lab(line)
        if matched is None:
            continue
        raw_name, canonical_name, rest = matched
        observation = _observation_from_match(
            raw_name=raw_name,
            canonical_name=canonical_name,
            rest=rest,
            source=source,
            date_lexeme=date_lexeme,
            observed_date=observed_date,
            date_flags=date_flags,
        )
        if observation is None or observation.evidence_id in seen:
            continue
        seen.add(observation.evidence_id)
        observations.append(observation)
    return observations


class EvidenceCompiler:
    """Compile a retrieval result into normalized, source-linked evidence."""

    def compile(
        self, *, retrieval: RetrievalResult, as_of: datetime | None = None
    ) -> EvidencePackage:
        if as_of is not None and as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        observations: list[ParsedObservation] = []
        excluded: list[ExcludedObservation] = []
        sources: list[EvidenceSource] = []
        seen: set[str] = set()
        for chunk in retrieval.items:
            source = EvidenceSource(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                content_hash=chunk.content_hash,
                method=chunk.method,
                quality=chunk.quality,
                document_created_at=chunk.document_created_at,
                score=chunk.score,
            )
            sources.append(source)
            for observation in parse_lab_observations(chunk.text, source=source):
                if observation.evidence_id in seen:
                    continue
                seen.add(observation.evidence_id)
                if (
                    as_of is not None
                    and observation.observed_date is not None
                    and observation.observed_date > as_of.date()
                ):
                    excluded.append(
                        ExcludedObservation(
                            evidence_id=observation.evidence_id,
                            canonical_name=observation.canonical_name,
                            observed_date=observation.observed_date,
                        )
                    )
                    continue
                observations.append(observation)
        findings, conflicts, missing_information = _summarize(observations)
        return EvidencePackage(
            query=retrieval.query,
            patient_id=retrieval.patient_id,
            as_of=as_of,
            retrieval_status=retrieval.status,
            filters=retrieval.filters,
            observations=observations,
            findings=findings,
            conflicts=conflicts,
            missing_information=missing_information,
            excluded_after_as_of=excluded,
            sources=sources,
        )


def _match_lab(line: str) -> tuple[str, str, str] | None:
    stripped = _BULLET_RE.sub("", line.strip())
    if not stripped:
        return None
    lowered = stripped.casefold()
    for alias in _ALIASES:
        if not lowered.startswith(alias):
            continue
        remainder = stripped[len(alias) :]
        if remainder and not (remainder[0].isspace() or remainder[0] in ":="):
            continue
        rest = _PARENTHETICAL_RE.sub("", remainder.lstrip())
        rest = rest.lstrip(" :=")
        if _VALUE_RE.match(rest) or _PENDING_RE.match(rest.rstrip(" .;,")):
            return stripped[: len(alias)].strip(), _CANONICAL_LABS[alias], rest
        return None
    return None


def _observation_from_match(
    *,
    raw_name: str,
    canonical_name: str,
    rest: str,
    source: EvidenceSource,
    date_lexeme: str | None,
    observed_date: date | None,
    date_flags: list[EvidenceFlag],
) -> ParsedObservation | None:
    if _PENDING_RE.match(rest.rstrip(" .;,")):
        return _build_observation(
            raw_name=raw_name,
            canonical_name=canonical_name,
            value_lexeme=None,
            unit_lexeme=None,
            range_lexeme=None,
            low_lexeme=None,
            high_lexeme=None,
            range_unit=None,
            outside=None,
            status=ObservationStatus.MISSING,
            source=source,
            date_lexeme=date_lexeme,
            observed_date=observed_date,
            extra_flags=date_flags,
        )
    value_match = _VALUE_RE.match(rest)
    if value_match is None:
        return None
    value_lexeme = value_match.group("value")
    remainder = rest[value_match.end() :]
    unit_lexeme, tail = _match_unit(remainder)
    range_lexeme, low_lexeme, high_lexeme, range_unit = _match_range(tail)
    flags = list(date_flags)
    status = ObservationStatus.EXPLICIT
    if source.quality == "uncertain":
        flags.append(EvidenceFlag.OCR_UNCERTAIN)
        status = ObservationStatus.UNCERTAIN
    if unit_lexeme is None:
        flags.append(EvidenceFlag.MISSING_UNIT)
        status = ObservationStatus.UNCERTAIN
    outside: bool | None = None
    if low_lexeme is not None and high_lexeme is not None:
        if range_unit is None:
            flags.append(EvidenceFlag.RANGE_UNIT_MISSING)
        elif unit_lexeme is not None and range_unit.casefold() == unit_lexeme.casefold():
            outside = Decimal(value_lexeme) < Decimal(low_lexeme) or Decimal(
                value_lexeme
            ) > Decimal(high_lexeme)
            if outside:
                flags.append(EvidenceFlag.OUTSIDE_REFERENCE_RANGE)
        elif unit_lexeme is not None:
            flags.append(EvidenceFlag.RANGE_UNIT_MISMATCH)
    return _build_observation(
        raw_name=raw_name,
        canonical_name=canonical_name,
        value_lexeme=value_lexeme,
        unit_lexeme=unit_lexeme,
        range_lexeme=range_lexeme,
        low_lexeme=low_lexeme,
        high_lexeme=high_lexeme,
        range_unit=range_unit,
        outside=outside,
        status=status,
        source=source,
        date_lexeme=date_lexeme,
        observed_date=observed_date,
        extra_flags=flags,
    )


def _match_unit(remainder: str) -> tuple[str | None, str]:
    match = _UNIT_RE.match(remainder)
    if match is None:
        return None, remainder
    unit = match.group("unit").rstrip(".,;:")
    if unit.casefold() in _UNIT_STOPWORDS:
        return None, remainder
    return unit, remainder[match.end() :]


def _match_range(tail: str) -> tuple[str | None, str | None, str | None, str | None]:
    for pattern in (_PARENTHESIZED_RANGE_RE, _REFERENCE_RANGE_RE):
        match = pattern.search(tail)
        if match is None:
            continue
        low, high = match.group("low"), match.group("high")
        if Decimal(low) > Decimal(high):
            continue
        range_unit = match.group("unit")
        if range_unit is not None:
            range_unit = range_unit.rstrip(".,;:")
            if range_unit.casefold() in _UNIT_STOPWORDS:
                range_unit = None
        return match.group(0).strip(), low, high, range_unit
    return None, None, None, None


def _build_observation(
    *,
    raw_name: str,
    canonical_name: str,
    value_lexeme: str | None,
    unit_lexeme: str | None,
    range_lexeme: str | None,
    low_lexeme: str | None,
    high_lexeme: str | None,
    range_unit: str | None,
    outside: bool | None,
    status: ObservationStatus,
    source: EvidenceSource,
    date_lexeme: str | None,
    observed_date: date | None,
    extra_flags: list[EvidenceFlag],
) -> ParsedObservation:
    identity = "|".join(
        [
            str(source.document_id),
            source.chunk_id,
            str(source.page_number),
            source.content_hash,
            canonical_name,
            raw_name,
            value_lexeme or "",
            unit_lexeme or "",
            range_lexeme or "",
            date_lexeme or "",
        ]
    )
    evidence_id = f"ev_{sha256(identity.encode('utf-8')).hexdigest()[:24]}"
    return ParsedObservation(
        evidence_id=evidence_id,
        raw_name=raw_name,
        canonical_name=canonical_name,
        value_lexeme=value_lexeme,
        value_decimal=Decimal(value_lexeme) if value_lexeme is not None else None,
        unit_lexeme=unit_lexeme,
        reference_range_lexeme=range_lexeme,
        reference_low_lexeme=low_lexeme,
        reference_high_lexeme=high_lexeme,
        reference_low=Decimal(low_lexeme) if low_lexeme is not None else None,
        reference_high=Decimal(high_lexeme) if high_lexeme is not None else None,
        reference_unit_lexeme=range_unit,
        outside_reference_range=outside,
        status=status,
        flags=extra_flags,
        observed_date_lexeme=date_lexeme,
        observed_date=observed_date,
        source=source,
    )


def _chunk_date(text: str) -> tuple[str | None, date | None, list[EvidenceFlag]]:
    lexemes = list(
        dict.fromkeys(
            [match.group(0) for match in _ISO_DATE_RE.finditer(text)]
            + [match.group(0) for match in _SLASH_DATE_RE.finditer(text)]
        )
    )
    if not lexemes:
        return None, None, []
    if len(lexemes) > 1:
        return None, None, [EvidenceFlag.DATE_AMBIGUOUS]
    try:
        parsed = date.fromisoformat(lexemes[0].replace("/", "-"))
    except ValueError:
        return lexemes[0], None, [EvidenceFlag.DATE_UNPARSEABLE]
    return lexemes[0], parsed, []


def _summarize(
    observations: list[ParsedObservation],
) -> tuple[list[LabFinding], list[LabConflict], list[str]]:
    grouped: dict[str, list[ParsedObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.canonical_name, []).append(observation)
    findings: list[LabFinding] = []
    conflicts: list[LabConflict] = []
    missing_information: list[str] = []
    for canonical_name in sorted(grouped):
        group = grouped[canonical_name]
        value_pairs = {
            (observation.value_decimal, (observation.unit_lexeme or "").casefold())
            for observation in group
            if observation.value_decimal is not None
        }
        evidence_ids = [observation.evidence_id for observation in group]
        if len(value_pairs) > 1:
            status = FindingStatus.CONFLICTING
            conflicts.append(
                LabConflict(
                    canonical_name=canonical_name,
                    value_lexemes=_ordered_unique(
                        observation.value_lexeme
                        for observation in group
                        if observation.value_lexeme is not None
                    ),
                    unit_lexemes=_ordered_unique(
                        observation.unit_lexeme
                        for observation in group
                        if observation.unit_lexeme is not None
                    ),
                    evidence_ids=evidence_ids,
                )
            )
        elif not value_pairs:
            status = FindingStatus.MISSING
            missing_information.append(canonical_name)
        elif all(
            observation.status is ObservationStatus.UNCERTAIN for observation in group
        ):
            status = FindingStatus.UNCERTAIN
        else:
            status = FindingStatus.EXPLICIT
        findings.append(
            LabFinding(canonical_name=canonical_name, status=status, evidence_ids=evidence_ids)
        )
    return findings, conflicts, missing_information


def _ordered_unique(values: Iterable[str | None]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value is not None))

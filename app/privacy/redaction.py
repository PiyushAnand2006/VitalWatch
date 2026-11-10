"""In-memory token vault and span redaction.

Tokens are stable within one payload so an external model can refer to evidence
references, while the original values never leave this process. The vault is not
persisted, so approval audit rows cannot reconstruct any redacted value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from app.privacy.engine import PiiAnalyzer, PiiEntity
from app.privacy.temporal import age_band, age_band_from_lexeme, parse_date_lexeme

_TOKEN_LABELS = {
    "ADDRESS": "ADDRESS",
    "AGE": "AGE",
    "CREDIT_CARD": "CARD",
    "DATE_OF_BIRTH": "DOB",
    "DATE_TIME": "DATE",
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "EVIDENCE": "EV",
    "FACILITY": "FACILITY",
    "GOVERNMENT_ID": "GOV_ID",
    "IBAN_CODE": "IBAN",
    "ID": "ID",
    "INTERNAL_ID": "INTERNAL_ID",
    "IP_ADDRESS": "IP",
    "LOCATION": "LOCATION",
    "MEDICAL_ID": "MEDICAL_ID",
    "MEDICAL_LICENSE": "MEDICAL_LICENSE",
    "MEDICAL_RECORD_NUMBER": "MRN",
    "NRP": "NRP",
    "ORGANIZATION": "ORG",
    "PERSON": "PERSON",
    "PHONE_NUMBER": "PHONE",
    "UK_NHS": "NHS",
    "URL": "URL",
    "US_BANK_NUMBER": "BANK",
    "US_DRIVER_LICENSE": "DL",
    "US_ITIN": "ITIN",
    "US_PASSPORT": "PASSPORT",
    "US_SSN": "SSN",
}


class TokenVault:
    """Stable per-payload tokens with an in-memory reverse map."""

    def __init__(self) -> None:
        self._tokens_by_value: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}
        self._reverse: dict[str, str] = {}

    def token_for(self, entity_type: str, value: str) -> str:
        """Return a stable token such as ``[PERSON_1]`` for one value."""

        key = (entity_type, value)
        existing = self._tokens_by_value.get(key)
        if existing is not None:
            return existing
        label = _TOKEN_LABELS.get(
            entity_type, re.sub(r"[^A-Z0-9_]", "_", entity_type.upper())
        )
        index = self._counters.get(label, 0) + 1
        self._counters[label] = index
        token = f"[{label}_{index}]"
        self._tokens_by_value[key] = token
        self._reverse[token] = value
        return token

    def resolve(self, token: str) -> str | None:
        """Resolve a token locally; the result is never persisted or logged."""

        return self._reverse.get(token)

    def contains_unredacted_value(self, text: str) -> bool:
        """Detect a redacted value that survived span replacement."""

        return any(
            len(value) >= 4 and value in text
            for value in self._reverse.values()
            if not value.startswith("ev_")
        )


@dataclass(frozen=True)
class RedactionResult:
    """Redacted text plus non-sensitive counts and review flags."""

    text: str
    counts: dict[str, int] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    entity_types: frozenset[str] = frozenset()


def redact_text(
    text: str,
    *,
    analyzer: PiiAnalyzer,
    vault: TokenVault,
    on_date: date,
) -> RedactionResult:
    """Replace direct identifiers and generalize dates/ages in untrusted text."""

    entities = _clamp(analyzer.analyze(text), len(text))
    entity_types = frozenset(entity.entity_type for entity in entities)
    counts: dict[str, int] = {}
    flags: list[str] = []

    replacements: list[tuple[int, int, str, str]] = []
    remaining = list(entities)

    dob_entities = [entity for entity in remaining if entity.entity_type == "DATE_OF_BIRTH"]
    if dob_entities:
        flags.append("dob_generalized")
        for entity in dob_entities:
            parsed = parse_date_lexeme(text[entity.start : entity.end])
            band = age_band(parsed, on_date) if parsed is not None else None
            if band is None:
                replacement = "[DOB_REDACTED]"
                flags.append("invalid_dob")
            else:
                replacement = f"[AGE_BAND_{band}]"
                if band == "90_plus":
                    flags.append("age_90_plus")
            replacements.append((entity.start, entity.end, replacement, "DATE_OF_BIRTH"))
            remaining = [
                candidate
                for candidate in remaining
                if candidate.start >= entity.end or candidate.end <= entity.start
            ]

    age_entities = [entity for entity in remaining if entity.entity_type == "AGE"]
    for entity in age_entities:
        band = age_band_from_lexeme(text[entity.start : entity.end])
        if band is None:
            replacement = "[AGE_REDACTED]"
        else:
            replacement = f"[AGE_BAND_{band}]"
            if band == "90_plus" and "age_90_plus" not in flags:
                flags.append("age_90_plus")
        replacements.append((entity.start, entity.end, replacement, "AGE"))
        remaining = [
            candidate
            for candidate in remaining
            if candidate.start >= entity.end or candidate.end <= entity.start
        ]

    for entity in _merge_spans(remaining):
        token = vault.token_for(entity.entity_type, text[entity.start : entity.end])
        replacements.append((entity.start, entity.end, token, entity.entity_type))

    replacements.sort(key=lambda item: (item[0], item[1]))
    pieces: list[str] = []
    cursor = 0
    for start, end, replacement, entity_type in replacements:
        if start < cursor:
            continue
        pieces.append(text[cursor:start])
        pieces.append(replacement)
        cursor = end
        counts[entity_type] = counts.get(entity_type, 0) + 1
    pieces.append(text[cursor:])

    return RedactionResult(
        text="".join(pieces),
        counts=counts,
        flags=list(dict.fromkeys(flags)),
        entity_types=entity_types,
    )


def _clamp(entities: list[PiiEntity], text_length: int) -> list[PiiEntity]:
    clamped: list[PiiEntity] = []
    for entity in entities:
        start = max(0, min(entity.start, text_length))
        end = max(start, min(entity.end, text_length))
        if end > start:
            clamped.append(
                PiiEntity(
                    entity_type=entity.entity_type,
                    start=start,
                    end=end,
                    score=entity.score,
                )
            )
    return clamped


def _merge_spans(entities: list[PiiEntity]) -> list[PiiEntity]:
    """Merge overlapping detections so no part of a match can leak between tokens."""

    ordered = sorted(entities, key=lambda entity: (entity.start, -(entity.end - entity.start)))
    merged: list[PiiEntity] = []
    for entity in ordered:
        if not merged or entity.start >= merged[-1].end:
            merged.append(entity)
            continue
        previous = merged[-1]
        merged[-1] = PiiEntity(
            entity_type=(
                entity.entity_type
                if (entity.end - entity.start) > (previous.end - previous.start)
                else previous.entity_type
            ),
            start=previous.start,
            end=max(previous.end, entity.end),
            score=max(previous.score, entity.score),
        )
    return merged

"""Shared deterministic extraction helpers for the standalone verifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass

ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?\b")
YEAR = re.compile(r"\b(\d{4})\b")
EVIDENCE_TOKEN = re.compile(r"\[[A-Z][A-Z0-9_]*_\d+\]")
# The lookbehind keeps range separators attached to a previous number ("3.5-5.1")
# from being read as a negative value.
NUMBER = re.compile(r"(?<![0-9.])[+-]?\d+(?:\.\d+)?")
UNIT = re.compile(r"[A-Za-z%µ°][A-Za-z0-9/%^*µ°·×-]*")

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
MONTH_NAME_DATE = re.compile(
    r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b", re.IGNORECASE
)

# English connectives that may follow a number but are never its unit.
_UNIT_STOP_WORDS = frozenset(
    {
        "and",
        "or",
        "to",
        "the",
        "a",
        "an",
        "of",
        "in",
        "is",
        "was",
        "were",
        "which",
        "that",
        "with",
        "for",
        "times",
        "fold",
        "day",
        "days",
        "hour",
        "hours",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "percent",
    }
)


@dataclass(frozen=True)
class MeasuredNumber:
    """One numeric lexeme extracted from claim text."""

    lexeme: str
    unit: str | None
    has_decimal: bool


def _unit_after(text: str, position: int) -> str | None:
    """Return the unit candidate directly after a number, or None.

    Only unit-shaped candidates (carrying /, %, µ, °, digits, or *) qualify;
    English words that merely follow a number are never treated as units.
    """

    rest = text[position:]
    stripped = rest.lstrip(" ")
    if len(rest) - len(stripped) > 1:  # more than one space before it: not a unit
        return None
    candidate = UNIT.match(stripped)
    if candidate is None:
        return None
    unit = candidate.group(0)
    if unit.isalpha():  # "mmol/L" and "5%" are units; "abnormal" is not
        return None
    if unit.lower() in _UNIT_STOP_WORDS:
        return None
    return unit


def extract_measured_numbers(text: str) -> list[MeasuredNumber]:
    """Extract number/unit pairs without interpreting dates or evidence tokens."""

    cleaned = ISO_DATE.sub(" ", text)
    cleaned = MONTH_NAME_DATE.sub(" ", cleaned)
    cleaned = YEAR.sub(" ", cleaned)
    cleaned = EVIDENCE_TOKEN.sub(" ", cleaned)

    measured: list[MeasuredNumber] = []
    for match in NUMBER.finditer(cleaned):
        lexeme = match.group(0)
        unit = _unit_after(cleaned, match.end())
        measured.append(
            MeasuredNumber(lexeme=lexeme, unit=unit, has_decimal="." in lexeme)
        )
    return measured


def extract_year_months(text: str) -> list[tuple[int, int]]:
    """Extract (year, month) pairs from ISO and month-name dates in claim text."""

    dates: list[tuple[int, int]] = []
    for match in ISO_DATE.finditer(text):
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            dates.append((year, month))
    for match in MONTH_NAME_DATE.finditer(text):
        month = MONTHS[match.group(1).lower()]
        dates.append((int(match.group(2)), month))
    return list(dict.fromkeys(dates))

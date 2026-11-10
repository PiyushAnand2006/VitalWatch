"""Conservative temporal parsing and generalization for outbound privacy payloads."""

from __future__ import annotations

import re
from datetime import date

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_ISO_OR_YEAR_FIRST = re.compile(
    r"\b(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})\b"
)
_US_SLASH = re.compile(r"\b(?P<first>\d{1,2})/(?P<second>\d{1,2})/(?P<year>\d{4})\b")
_MONTH_NAME_FIRST = re.compile(
    r"\b(?P<month>[A-Za-z]{3,9})\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})\b"
)
_DAY_MONTH_NAME = re.compile(
    r"\b(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]{3,9})\.?,?\s+(?P<year>\d{4})\b"
)


def parse_date_lexeme(value: str) -> date | None:
    """Parse only unambiguous date forms; ambiguous numeric dates fail closed."""

    match = _ISO_OR_YEAR_FIRST.search(value)
    if match is not None:
        return _safe_date(
            int(match.group("year")), int(match.group("month")), int(match.group("day"))
        )

    match = _MONTH_NAME_FIRST.search(value)
    if match is not None:
        month = _MONTHS.get(match.group("month").casefold())
        if month is not None:
            return _safe_date(int(match.group("year")), month, int(match.group("day")))

    match = _DAY_MONTH_NAME.search(value)
    if match is not None:
        month = _MONTHS.get(match.group("month").casefold())
        if month is not None:
            return _safe_date(int(match.group("year")), month, int(match.group("day")))

    match = _US_SLASH.search(value)
    if match is not None:
        first, second = int(match.group("first")), int(match.group("second"))
        if first <= 12 and second <= 12:
            # Month/day order is unknowable without a locale; do not guess.
            return None
        if first > 12 and second <= 12:
            return _safe_date(int(match.group("year")), second, first)
    return None


def age_band(birth_date: date, on_date: date) -> str | None:
    """Return a coarse age band, or None for an implausible birth date."""

    age = on_date.year - birth_date.year
    if (on_date.month, on_date.day) < (birth_date.month, birth_date.day):
        age -= 1
    if age < 0 or age > 130:
        return None
    if age >= 90:
        return "90_plus"
    lower = (age // 10) * 10
    return f"{lower}_{lower + 9}"


def age_band_from_lexeme(value: str) -> str | None:
    """Generalize an explicit age to the same bands used for date of birth."""

    match = re.search(r"\d{1,3}", value)
    if match is None:
        return None
    age = int(match.group(0))
    if age < 0 or age > 130:
        return None
    if age >= 90:
        return "90_plus"
    lower = (age // 10) * 10
    return f"{lower}_{lower + 9}"


def generalize_observed_date(value: date | None) -> str | None:
    """Reduce an observation date to month precision; never send a full date."""

    if value is None:
        return None
    return f"{value.year:04d}-{value.month:02d}"


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None

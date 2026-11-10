"""Shared Pydantic annotated types used across schema modules."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import AfterValidator


def _require_timezone(value: datetime) -> datetime:
    """Reject naive datetimes so persisted timestamps stay unambiguous."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value


AwareDatetime = Annotated[datetime, AfterValidator(_require_timezone)]

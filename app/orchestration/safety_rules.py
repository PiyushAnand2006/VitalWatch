"""Configurable deterministic urgent-warning rules applied before any debate."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from app.schemas.orchestration import UrgentRule
from app.schemas.privacy import PreparedPayload


@dataclass(frozen=True)
class UrgentFinding:
    """One triggered warning sign, described without patient data."""

    label: str
    observation_ref: str


def evaluate_urgent_rules(
    payload: PreparedPayload, rules: tuple[UrgentRule, ...]
) -> list[UrgentFinding]:
    """Evaluate every configured rule against the minimized observations."""

    findings: list[UrgentFinding] = []
    for rule in rules:
        pattern = re.compile(rule.lab_pattern, re.IGNORECASE)
        threshold = _decimal(rule.value)
        if threshold is None:
            continue
        for observation in payload.observations:
            if observation.value is None:
                continue
            if pattern.search(observation.lab) is None:
                continue
            if rule.unit is not None and observation.unit != rule.unit:
                continue
            value = _decimal(observation.value)
            if value is None:
                continue
            if _compare(value, rule.operator, threshold):
                findings.append(
                    UrgentFinding(label=rule.label, observation_ref=observation.ref)
                )
    return findings


def _compare(value: Decimal, operator: str, threshold: Decimal) -> bool:
    if operator == "gt":
        return value > threshold
    if operator == "gte":
        return value >= threshold
    if operator == "lt":
        return value < threshold
    return value <= threshold


def _decimal(lexeme: str) -> Decimal | None:
    try:
        return Decimal(lexeme)
    except InvalidOperation:
        return None

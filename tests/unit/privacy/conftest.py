"""Shared real-Presidio fixtures for privacy unit tests."""

from datetime import date

import pytest

from app.privacy.engine import PresidioPiiAnalyzer


@pytest.fixture(scope="module")
def presidio() -> PresidioPiiAnalyzer:
    return PresidioPiiAnalyzer()


@pytest.fixture
def on_date() -> date:
    return date(2026, 11, 10)

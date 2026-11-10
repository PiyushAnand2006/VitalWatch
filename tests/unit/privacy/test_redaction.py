"""Real Presidio redaction, tokenization, and temporal generalization tests."""

from datetime import date

from app.privacy.engine import PresidioPiiAnalyzer
from app.privacy.redaction import TokenVault, redact_text
from app.privacy.temporal import age_band, generalize_observed_date, parse_date_lexeme


def test_varied_identifiers_are_redacted_without_raw_values(
    presidio: PresidioPiiAnalyzer, on_date: date
) -> None:
    text = (
        "John Smith (MRN: AB-123456), phone 555-123-4567, email john@example.com, "
        "SSN 123-45-6789, Aadhaar 2345 6789 0123, passport X1234567, "
        "insurance policy ID: POL-998877, patient id: patient-123, "
        "123 Main Street, Mercy Hospital, DOB 1940-01-01."
    )

    result = redact_text(text, analyzer=presidio, vault=TokenVault(), on_date=on_date)

    for raw in (
        "John Smith",
        "AB-123456",
        "555-123-4567",
        "john@example.com",
        "123-45-6789",
        "2345 6789 0123",
        "X1234567",
        "POL-998877",
        "patient-123",
        "123 Main Street",
        "Mercy Hospital",
        "1940-01-01",
    ):
        assert raw not in result.text
    assert result.counts["PERSON"] == 1
    assert result.counts["MEDICAL_RECORD_NUMBER"] == 1
    assert result.flags == ["dob_generalized"]
    assert "[AGE_BAND_80_89]" in result.text


def test_internal_and_evidence_identifiers_are_tokenized(
    presidio: PresidioPiiAnalyzer, on_date: date
) -> None:
    text = (
        "document id 8f14e45f-ceea-467a-9d1b-1c2d3e4f5a6b "
        "and ev_0123456789abcdef01234567"
    )

    result = redact_text(text, analyzer=presidio, vault=TokenVault(), on_date=on_date)

    assert "8f14e45f-ceea-467a-9d1b-1c2d3e4f5a6b" not in result.text
    assert "ev_0123456789abcdef01234567" not in result.text
    assert result.counts["INTERNAL_ID"] == 2


def test_malformed_dob_fails_closed(
    presidio: PresidioPiiAnalyzer, on_date: date
) -> None:
    result = redact_text(
        "DOB 99/99/9999", analyzer=presidio, vault=TokenVault(), on_date=on_date
    )

    assert "99/99/9999" not in result.text
    assert result.text == "[DOB_REDACTED]"
    assert "invalid_dob" in result.flags


def test_ambiguous_numeric_dob_is_not_guessed(
    presidio: PresidioPiiAnalyzer, on_date: date
) -> None:
    result = redact_text(
        "DOB 01/02/1940", analyzer=presidio, vault=TokenVault(), on_date=on_date
    )

    assert "01/02/1940" not in result.text
    assert result.text == "[DOB_REDACTED]"
    assert "invalid_dob" in result.flags


def test_age_90_plus_is_generalized_and_flagged(
    presidio: PresidioPiiAnalyzer, on_date: date
) -> None:
    result = redact_text(
        "age 92-year-old", analyzer=presidio, vault=TokenVault(), on_date=on_date
    )

    assert result.text == "[AGE_BAND_90_plus]"
    assert "age_90_plus" in result.flags


def test_date_time_is_tokenized_instead_of_transmitted(
    presidio: PresidioPiiAnalyzer, on_date: date
) -> None:
    result = redact_text(
        "Visit on 2026-01-05", analyzer=presidio, vault=TokenVault(), on_date=on_date
    )

    assert "2026-01-05" not in result.text
    assert "[DATE_1]" in result.text


def test_token_vault_is_stable_and_reversible_only_in_memory() -> None:
    vault = TokenVault()
    first = vault.token_for("PERSON", "John Smith")
    second = vault.token_for("PERSON", "John Smith")

    assert first == second == "[PERSON_1]"
    assert vault.resolve(first) == "John Smith"
    assert vault.contains_unredacted_value("[PERSON_1]") is False


def test_observation_dates_are_reduced_to_month_precision() -> None:
    assert generalize_observed_date(date(2026, 1, 5)) == "2026-01"
    assert generalize_observed_date(None) is None


def test_dob_parsing_and_age_bands_are_conservative() -> None:
    assert parse_date_lexeme("DOB 1940-01-01") == date(1940, 1, 1)
    assert parse_date_lexeme("DOB 01/02/1940") is None
    assert age_band(date(1940, 1, 1), date(2026, 11, 10)) == "80_89"
    assert age_band(date(1900, 1, 1), date(2026, 11, 10)) == "90_plus"
    assert age_band(date(2030, 1, 1), date(2026, 11, 10)) is None

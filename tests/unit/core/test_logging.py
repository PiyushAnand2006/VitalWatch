"""Sensitive-value and request-context logging tests."""

import logging
import sys

import pytest

from app.core.logging import RedactingFormatter, RequestIDFilter, redact_text


@pytest.mark.parametrize(
    ("message", "sensitive_value"),
    [
        ("api_key=super-secret-api-value", "super-secret-api-value"),
        ("Authorization: Bearer abc.def.ghi", "abc.def.ghi"),
        ("contact=synthetic.patient@example.org", "synthetic.patient@example.org"),
        ("MRN: SYN-12345", "SYN-12345"),
        ("credential sk-abcdefghijklmnop", "sk-abcdefghijklmnop"),
    ],
)
def test_redact_text_removes_sensitive_values(message: str, sensitive_value: str) -> None:
    redacted = redact_text(message)

    assert sensitive_value not in redacted
    assert "REDACTED" in redacted


def test_formatter_redacts_exception_tracebacks() -> None:
    logger = logging.getLogger("stage1-test")
    try:
        raise ValueError("api_key=traceback-secret-value")
    except ValueError:
        record = logger.makeRecord(
            logger.name,
            logging.ERROR,
            __file__,
            1,
            "Synthetic failure",
            (),
            exc_info=sys.exc_info(),
        )

    rendered = RedactingFormatter("%(levelname)s %(message)s").format(record)

    assert "traceback-secret-value" not in rendered
    assert "REDACTED" in rendered


def test_request_id_filter_supplies_non_request_default() -> None:
    record = logging.LogRecord("stage1-test", logging.INFO, __file__, 1, "message", (), None)

    assert RequestIDFilter().filter(record) is True
    assert record.request_id == "system"

"""Request-aware logging with defense-in-depth sensitive-value redaction."""

import logging
import logging.config
import re
from typing import Any

from app.core.config import Settings
from app.core.request_context import get_request_id

_REDACTIONS = (
    (
        re.compile(
            r"(?i)\b(authorization|api[_-]?key|access[_-]?token|auth[_-]?token|"
            r"secret[_-]?key|password)\b(\s*[:=]\s*)"
                        r"(\"[^\"]*\"|'[^']*'|Bearer\s+[^\s,;]+|[^\s,;]+)"
        ),
        r"\1\2[REDACTED]",
    ),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "[REDACTED_API_KEY]"),
    (
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(
            r"(?i)\b(mrn|medical[ _-]?record[ _-]?number|patient[ _-]?id)\b"
            r"(\s*[:=#-]\s*)([A-Za-z0-9-]+)"
        ),
        r"\1\2[REDACTED_ID]",
    ),
)


def redact_text(value: str) -> str:
    """Redact common secret and identifier forms from a rendered log message."""

    redacted = value
    for pattern, replacement in _REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


class RequestIDFilter(logging.Filter):
    """Make the active request ID available to the formatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


class RedactingFormatter(logging.Formatter):
    """Redact the complete rendered record, including exception tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_text(super().format(record))


def configure_logging(settings: Settings) -> None:
    """Configure process logging without message bodies or sensitive payloads."""

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "request_id": {"()": "app.core.logging.RequestIDFilter"},
        },
        "formatters": {
            "safe": {
                "()": "app.core.logging.RedactingFormatter",
                "format": (
                    "%(asctime)s %(levelname)s %(name)s "
                    "request_id=%(request_id)s %(message)s"
                ),
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
                "formatter": "safe",
                "filters": ["request_id"],
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["console"],
        },
        "loggers": {
            "uvicorn": {"level": settings.log_level},
            "uvicorn.error": {
                "level": settings.log_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": settings.log_level,
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }
    logging.config.dictConfig(config)

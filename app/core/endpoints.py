"""Shared endpoint validation: TLS for remote hosts, loopback plaintext allowed."""

from __future__ import annotations

from urllib.parse import urlsplit

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def validate_remote_endpoint(value: str | None) -> str | None:
    """Validate a configured external endpoint.

    Rejects non-absolute URLs, plaintext remote hosts, embedded credentials,
    and query strings or fragments. Returns the value unchanged when valid;
    ``None`` passes through so optional settings can stay unset.
    """

    if value is None:
        return None
    parsed = urlsplit(value)
    loopback = parsed.hostname in LOOPBACK_HOSTS
    if not parsed.hostname or (
        parsed.scheme != "https" and not (parsed.scheme == "http" and loopback)
    ):
        raise ValueError("endpoint must use https (http is loopback-only)")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint must not contain credentials")
    if parsed.query:
        raise ValueError("endpoint must not contain a query string")
    if parsed.fragment:
        raise ValueError("endpoint must not contain a fragment")
    return value

"""Canonical serialization and SHA-256 bindings for privacy approvals."""

from __future__ import annotations

import json
from hashlib import sha256

from pydantic import BaseModel

from app.schemas.privacy import ExternalDestination, PreparedPayload


def canonical_json(value: BaseModel) -> str:
    """Serialize a contract deterministically across processes and runs."""

    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def payload_sha256(payload: PreparedPayload) -> str:
    """Bind an approval to the exact serialized outbound payload."""

    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def destination_sha256(destination: ExternalDestination) -> str:
    """Bind an approval to one provider, endpoint, model, and purpose."""

    return sha256(canonical_json(destination).encode("utf-8")).hexdigest()


def stable_ref(value: str) -> str:
    """Hash an internal identifier before it enters a privacy audit record."""

    return sha256(value.encode("utf-8")).hexdigest()

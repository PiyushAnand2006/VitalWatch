"""Minimal HS256 JWT validation and patient-scope authorization dependencies."""

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.exceptions import AppError

_bearer = HTTPBearer(auto_error=False)


class AuthenticatedUser(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(min_length=1, max_length=128)


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def decode_access_token(token: str, secret: bytes) -> AuthenticatedUser:
    """Validate a deliberately small HS256 JWT contract (sub and exp required)."""

    try:
        header_segment, payload_segment, signature_segment = token.split(".")
        signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
        expected = hmac.new(secret, signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64decode(signature_segment)):
            raise ValueError("signature")
        header = json.loads(_b64decode(header_segment))
        payload = json.loads(_b64decode(payload_segment))
        if header != {"alg": "HS256", "typ": "JWT"} or not isinstance(payload, dict):
            raise ValueError("header")
        subject, expires_at = payload.get("sub"), payload.get("exp")
        if not isinstance(subject, str) or not isinstance(expires_at, int):
            raise ValueError("claims")
        if expires_at <= int(datetime.now(UTC).timestamp()):
            raise ValueError("expired")
        return AuthenticatedUser(user_id=subject)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, ValidationError):
        raise AppError(401, "invalid_token", "Authentication credentials are invalid.") from None


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AppError(401, "authentication_required", "Authentication is required.")
    secret = request.app.state.settings.app_secret_key.get_secret_value().encode("utf-8")
    return decode_access_token(credentials.credentials, secret)

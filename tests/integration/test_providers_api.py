"""The /api/v1/providers route exposes configuration status and never secrets."""

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.application import create_app
from app.core.config import AppEnvironment, Settings

pytestmark = pytest.mark.integration

SECRET = "stage7-providers-api-synthetic-secret-32-chars"
GOOGLE_KEY = "synthetic-google-key-not-real"
NVIDIA_KEY = "synthetic-nvidia-key-not-real"


def _bearer(subject: str = "synthetic-provider-user") -> str:
    def encode(value: dict[str, object]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    header = encode({"alg": "HS256", "typ": "JWT"})
    claims = encode({"sub": subject, "exp": int(time.time()) + 300})
    signing_input = f"{header}.{claims}"
    signature = hmac.new(SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode('ascii')}"


def make_client(tmp_path: Path, **overrides: object) -> TestClient:
    settings = Settings(
        _env_file=None,
        app_env=AppEnvironment.TEST,
        app_secret_key=SECRET,
        document_storage_directory=tmp_path / "documents",
        embedding_model_directory=tmp_path / "models",
        **overrides,  # type: ignore[arg-type]
    )
    app = create_app(settings)
    return TestClient(app)


def test_provider_status_requires_authentication(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        assert client.get("/api/v1/providers").status_code == 401
        spoofed = client.get(
            "/api/v1/providers",
            headers={"Authorization": "Bearer not-a-valid-token"},
        )
        assert spoofed.status_code == 401


def test_provider_status_reports_configuration_without_secrets(tmp_path: Path) -> None:
    with make_client(
        tmp_path,
        google_api_key=GOOGLE_KEY,
        nvidia_api_key=NVIDIA_KEY,
        llm_reasoner_a_provider="google",
        llm_reasoner_a_model="synthetic-google-reasoner",
        llm_adjudicator_provider="nvidia",
        llm_adjudicator_model="synthetic-nvidia-adjudicator",
    ) as client:
        response = client.get(
            "/api/v1/providers", headers={"Authorization": f"Bearer {_bearer()}"}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["external_llm_enabled"] is False
    assert body["local_only"] is True
    assert body["fallback_enabled"] is False
    assert body["fallback_requires_fresh_approval"] is True
    assert body["prompt_version"] == "v1"

    providers = {provider["provider"]: provider for provider in body["providers"]}
    assert providers["google"]["configured"] is True
    assert providers["google"]["api_key_configured"] is True
    assert providers["nvidia"]["configured"] is True
    assert providers["llama"]["configured"] is False

    slots = {slot["slot"]: slot for slot in body["slots"]}
    assert set(slots) == {
        "reasoner_a",
        "reasoner_b",
        "reasoner_c",
        "final_adjudicator",
        "fallback",
    }
    assert slots["reasoner_a"]["configured"] is True
    assert slots["reasoner_a"]["enabled"] is False  # external LLM disabled
    assert slots["reasoner_a"]["provider"] == "google"
    assert slots["reasoner_a"]["model"] == "synthetic-google-reasoner"
    assert slots["reasoner_b"]["configured"] is False
    assert slots["reasoner_b"]["enabled"] is False
    assert slots["final_adjudicator"]["configured"] is True

    text = response.text
    assert GOOGLE_KEY not in text
    assert NVIDIA_KEY not in text


def test_default_configuration_reports_every_slot_unconfigured(tmp_path: Path) -> None:
    with make_client(tmp_path) as client:
        response = client.get(
            "/api/v1/providers", headers={"Authorization": f"Bearer {_bearer()}"}
        )

    assert response.status_code == 200
    body = response.json()
    assert all(provider["configured"] is False for provider in body["providers"])
    assert all(slot["configured"] is False for slot in body["slots"])
    assert all(slot["enabled"] is False for slot in body["slots"])

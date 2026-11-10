"""Typed settings and secure-default tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import AppEnvironment, Settings
from tests.conftest import TEST_SECRET


def test_secure_defaults() -> None:
    settings = Settings(_env_file=None, app_secret_key=TEST_SECRET)

    assert settings.debug is False
    assert settings.api_docs_enabled is False
    assert settings.external_llm_enabled is False
    assert settings.privacy_gate_enabled is True


def test_secret_is_required() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "placeholder",
    [
        "replace_me",
        "change_me",
        "changeme",
        "replace_with_secure_random_secret_at_least_32_chars",
    ],
)
def test_known_placeholder_secrets_are_rejected(placeholder: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_secret_key=placeholder)


def test_short_secret_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_secret_key="too-short")


def test_privacy_gate_cannot_be_disabled() -> None:
    with pytest.raises(ValidationError, match="cannot be false"):
        Settings(
            _env_file=None,
            app_secret_key=TEST_SECRET,
            privacy_gate_enabled=False,
        )


def test_production_debug_is_rejected() -> None:
    with pytest.raises(ValidationError, match="DEBUG cannot be enabled"):
        Settings(
            _env_file=None,
            app_env=AppEnvironment.PRODUCTION,
            app_secret_key=TEST_SECRET,
            debug=True,
        )


def test_environment_values_are_loaded_as_typed_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_SECRET_KEY", TEST_SECRET)
    monkeypatch.setenv("API_DOCS_ENABLED", "true")
    monkeypatch.setenv("EXTERNAL_LLM_ENABLED", "false")
    monkeypatch.setenv("DOCUMENT_STORAGE_DIRECTORY", str(tmp_path / "private-docs"))

    settings = Settings(_env_file=None)

    assert settings.app_env == AppEnvironment.TEST
    assert settings.api_docs_enabled is True
    assert settings.external_llm_enabled is False
    assert settings.document_storage_directory == tmp_path / "private-docs"

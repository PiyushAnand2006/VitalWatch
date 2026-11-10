"""Shared synthetic Stage 1 test fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.application import create_app
from app.core.config import AppEnvironment, Settings

TEST_SECRET = "stage1-synthetic-test-secret-with-at-least-32-characters"
_ENVIRONMENT_VARIABLES = (
    "APP_ENV",
    "APP_NAME",
    "APP_VERSION",
    "APP_SECRET_KEY",
    "DEBUG",
    "API_DOCS_ENABLED",
    "LOG_LEVEL",
    "DATABASE_URL",
    "DOCUMENT_STORAGE_DIRECTORY",
    "DOCUMENT_STORAGE_ENCRYPTION_KEY",
    "UPLOAD_MAX_MB",
    "UPLOAD_MAX_PAGES",
    "PRIVACY_GATE_ENABLED",
    "EXTERNAL_LLM_ENABLED",
)


@pytest.fixture(autouse=True)
def clear_application_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host environment values from making tests non-deterministic."""

    for name in _ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        app_env=AppEnvironment.TEST,
        app_secret_key=TEST_SECRET,
        document_storage_directory=tmp_path / "documents",
        database_url="postgresql+psycopg://postgres:postgres@localhost:5432/vitalwatch_test",
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client

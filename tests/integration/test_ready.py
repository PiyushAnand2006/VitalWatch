"""Readiness success and failure integration tests."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.application import create_app
from app.core.config import AppEnvironment, Settings

SECRET = "stage5-readiness-synthetic-secret-at-least-32-characters"


def test_ready_reports_unavailable_database_and_lazy_model(
    client: TestClient,
    settings: Settings,
) -> None:
    response = client.get("/api/v1/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    statuses = {check["name"]: check["status"] for check in body["checks"]}
    assert statuses == {
        "configuration": "passed",
        "document_storage": "passed",
        "database": "failed",
        "embedding_model": "skipped",
    }
    assert "postgresql" not in response.text
    assert str(settings.document_storage_directory) not in response.text
    assert "X-Request-ID" in response.headers


def test_ready_returns_safe_503_when_storage_is_unavailable(
    client: TestClient,
    settings: Settings,
) -> None:
    settings.document_storage_directory.rmdir()
    settings.document_storage_directory.write_text("not a directory", encoding="utf-8")

    response = client.get("/api/v1/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    document_check = next(check for check in body["checks"] if check["name"] == "document_storage")
    assert document_check["status"] == "failed"
    assert str(settings.document_storage_directory) not in response.text
    assert "X-Request-ID" in response.headers


@pytest.mark.integration
def test_ready_passes_with_a_real_database(postgres_uri: str, tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env=AppEnvironment.TEST,
        app_secret_key=SECRET,
        database_url=postgres_uri.replace("postgresql://", "postgresql+psycopg://", 1),
        document_storage_directory=tmp_path / "documents",
        embedding_model_directory=tmp_path / "models",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/api/v1/ready")

    assert response.status_code == 200, response.text
    body = response.json()
    statuses = {check["name"]: check["status"] for check in body["checks"]}
    assert statuses == {
        "configuration": "passed",
        "document_storage": "passed",
        "database": "passed",
        "embedding_model": "skipped",
    }

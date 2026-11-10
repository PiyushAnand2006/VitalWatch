"""Readiness success and failure integration tests."""

from fastapi.testclient import TestClient

from app.core.config import Settings


def test_ready_checks_current_stage_dependencies(
    client: TestClient,
    settings: Settings,
) -> None:
    response = client.get("/api/v1/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert {check["name"] for check in body["checks"]} == {
        "configuration",
        "document_storage",
    }
    assert all(check["status"] == "passed" for check in body["checks"])
    assert settings.document_storage_directory.is_dir()
    assert str(settings.document_storage_directory) not in response.text


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

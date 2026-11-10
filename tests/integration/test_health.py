"""Liveness, request ID, and application-factory integration tests."""

from uuid import UUID

from fastapi.testclient import TestClient

from app.core.application import create_app
from app.core.config import Settings
from app.core.request_context import REQUEST_ID_HEADER


def test_health_returns_liveness_and_request_id(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "VitalWatch Part 2"
    assert body["environment"] == "test"
    assert body["request_id"] == response.headers[REQUEST_ID_HEADER]
    UUID(body["request_id"])


def test_valid_incoming_request_id_is_propagated(client: TestClient) -> None:
    response = client.get(
        "/api/v1/health",
        headers={REQUEST_ID_HEADER: "synthetic-request:001"},
    )

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] == "synthetic-request:001"
    assert response.json()["request_id"] == "synthetic-request:001"


def test_invalid_incoming_request_id_is_replaced(client: TestClient) -> None:
    response = client.get(
        "/api/v1/health",
        headers={REQUEST_ID_HEADER: "unsafe request id with spaces"},
    )

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] != "unsafe request id with spaces"
    UUID(response.headers[REQUEST_ID_HEADER])


def test_openapi_is_disabled_by_default(client: TestClient) -> None:
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_openapi_can_be_explicitly_enabled(settings: Settings) -> None:
    app = create_app(settings.model_copy(update={"api_docs_enabled": True}))

    with TestClient(app) as docs_client:
        assert docs_client.get("/docs").status_code == 200
        assert docs_client.get("/openapi.json").status_code == 200

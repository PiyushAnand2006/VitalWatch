"""Safe exception-envelope integration tests."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.application import create_app
from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.request_context import REQUEST_ID_HEADER


def _app_with_test_routes(settings: Settings) -> FastAPI:
    app = create_app(settings)

    @app.get("/_test/internal-error")
    async def internal_error() -> None:
        raise RuntimeError("api_key=must-not-leak-to-client")

    @app.get("/_test/expected-error")
    async def expected_error() -> None:
        raise AppError(409, "synthetic_conflict", "A safe conflict message.")

    @app.get("/_test/validated")
    async def validated(required: int) -> dict[str, int]:
        return {"required": required}

    return app


def test_unexpected_error_is_sanitized_and_has_request_id(settings: Settings) -> None:
    with TestClient(
        _app_with_test_routes(settings),
        raise_server_exceptions=False,
    ) as client:
        response = client.get("/_test/internal-error")

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "An internal error occurred.",
            "issues": None,
        },
        "request_id": response.headers[REQUEST_ID_HEADER],
    }
    assert "must-not-leak" not in response.text
    assert "RuntimeError" not in response.text
    assert "Traceback" not in response.text


def test_expected_application_error_uses_safe_public_message(settings: Settings) -> None:
    with TestClient(_app_with_test_routes(settings)) as client:
        response = client.get("/_test/expected-error")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "synthetic_conflict"
    assert response.json()["error"]["message"] == "A safe conflict message."


def test_validation_error_excludes_submitted_value_and_context(settings: Settings) -> None:
    with TestClient(_app_with_test_routes(settings)) as client:
        response = client.get("/_test/validated", params={"required": "not-an-integer"})

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["issues"][0]["location"] == ["query", "required"]
    assert "not-an-integer" not in response.text


def test_not_found_uses_safe_error_envelope(client: TestClient) -> None:
    response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["request_id"] == response.headers[REQUEST_ID_HEADER]

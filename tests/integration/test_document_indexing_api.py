"""End-to-end API verification that a processed upload is embedded and indexed."""

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from huggingface_hub.constants import HF_HUB_CACHE

from app.core.application import create_app
from app.core.config import AppEnvironment, Settings

pytestmark = pytest.mark.integration

SECRET = "stage4-api-synthetic-secret-with-at-least-32-characters"
PATIENT_ID = "synthetic-patient-a"
USER_ID = "synthetic-user-a"


def _bearer(subject: str = USER_ID) -> str:
    def encode(value: dict[str, object]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")

    header = encode({"alg": "HS256", "typ": "JWT"})
    claims = encode({"sub": subject, "exp": int(time.time()) + 300})
    signing_input = f"{header}.{claims}"
    signature = hmac.new(SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode('ascii')}"


def _grant_access(connection: psycopg.Connection[tuple[object, ...]]) -> None:
    connection.execute(
        "insert into patient_grants (id, user_id, patient_id, created_at) "
        "values (%s, %s, %s, now())",
        (uuid4(), USER_ID, PATIENT_ID),
    )


def test_authorized_upload_is_processed_and_indexed(
    postgres_uri: str,
    postgres_connection: psycopg.Connection[tuple[object, ...]],
    tmp_path: Path,
) -> None:
    _grant_access(postgres_connection)
    settings = Settings(
        _env_file=None,
        app_env=AppEnvironment.TEST,
        app_secret_key=SECRET,
        database_url=postgres_uri.replace("postgresql://", "postgresql+psycopg://", 1),
        document_storage_directory=tmp_path / "documents",
        embedding_model_directory=Path(HF_HUB_CACHE),
        embedding_batch_size=8,
    )
    app = create_app(settings)
    headers = {"Authorization": f"Bearer {_bearer()}"}

    with TestClient(app) as client:
        upload = client.post(
            f"/api/v1/patients/{PATIENT_ID}/documents",
            headers=headers,
            files={
                "file": (
                    "synthetic-note.txt",
                    b"Progress Note:\nPatient reports chest pain and shortness of breath.",
                    "text/plain",
                )
            },
        )
        assert upload.status_code == 201, upload.text
        document_id = upload.json()["id"]

        status = client.get(
            f"/api/v1/patients/{PATIENT_ID}/documents/{document_id}", headers=headers
        )
        assert status.status_code == 200, status.text
        assert status.json()["status"] == "processed"

    stored = postgres_connection.execute(
        "select count(*), min(dimension), min(model_name) from chunk_embeddings "
        "where patient_id = %s and document_id = %s",
        (PATIENT_ID, document_id),
    ).fetchone()
    assert stored is not None
    assert stored[0] >= 1
    assert stored[1] == 384
    assert stored[2] == "sentence-transformers/all-MiniLM-L6-v2"


def test_reprocess_reindexes_and_delete_removes_vectors(
    postgres_uri: str,
    postgres_connection: psycopg.Connection[tuple[object, ...]],
    tmp_path: Path,
) -> None:
    _grant_access(postgres_connection)
    settings = Settings(
        _env_file=None,
        app_env=AppEnvironment.TEST,
        app_secret_key=SECRET,
        database_url=postgres_uri.replace("postgresql://", "postgresql+psycopg://", 1),
        document_storage_directory=tmp_path / "documents",
        embedding_model_directory=Path(HF_HUB_CACHE),
        embedding_batch_size=8,
    )
    app = create_app(settings)
    headers = {"Authorization": f"Bearer {_bearer()}"}

    with TestClient(app) as client:
        upload = client.post(
            f"/api/v1/patients/{PATIENT_ID}/documents",
            headers=headers,
            files={
                "file": (
                    "synthetic-note.txt",
                    b"Laboratory Results:\nSodium 140 mmol/L\nPotassium 4.1 mmol/L",
                    "text/plain",
                )
            },
        )
        assert upload.status_code == 201, upload.text
        document_id = upload.json()["id"]
        reprocessed = client.post(
            f"/api/v1/patients/{PATIENT_ID}/documents/{document_id}/reprocess",
            headers=headers,
        )
        assert reprocessed.status_code == 200, reprocessed.text

        deleted = client.delete(
            f"/api/v1/patients/{PATIENT_ID}/documents/{document_id}", headers=headers
        )
        assert deleted.status_code == 204

    remaining = postgres_connection.execute(
        "select count(*) from chunk_embeddings where patient_id = %s and document_id = %s",
        (PATIENT_ID, document_id),
    ).fetchone()
    assert remaining is not None
    assert remaining[0] == 0

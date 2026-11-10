"""Tests for database configuration, session manager, and document storage boundaries."""

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.database import Base, DatabaseSessionManager, get_database_manager
from app.services.document_storage import LocalDocumentStorage
from tests.conftest import TEST_SECRET


def test_database_url_postgresql_default() -> None:
    settings = Settings(_env_file=None, app_secret_key=TEST_SECRET)
    assert settings.database_url.startswith("postgresql+psycopg://")


def test_database_manager_engine_initialization() -> None:
    settings = Settings(
        _env_file=None,
        app_secret_key=TEST_SECRET,
        database_url="postgresql+psycopg://user:pass@localhost:5432/vitalwatch_db",
    )
    db_manager = get_database_manager(settings)
    assert db_manager.engine is not None
    assert str(db_manager.engine.url).startswith("postgresql+psycopg://user:***@localhost:5432/vitalwatch_db")


def test_database_manager_converts_standard_postgresql_scheme() -> None:
    db_manager = DatabaseSessionManager(database_url="postgresql://user:pass@localhost:5432/vitalwatch_db")
    assert str(db_manager.engine.url).startswith("postgresql+psycopg://user:***@localhost:5432/vitalwatch_db")


def test_declarative_base_is_available() -> None:
    assert Base.metadata is not None


def test_local_document_storage_crud(tmp_path: Path) -> None:
    storage = LocalDocumentStorage(root_dir=tmp_path / "test_docs")
    doc_key = "report_123.txt"
    content = b"Medical report content payload"

    # Store
    file_path = storage.store(doc_key, content)
    assert Path(file_path).exists()
    assert storage.exists(doc_key) is True

    # Retrieve
    retrieved = storage.retrieve(doc_key)
    assert retrieved == content

    # Delete
    deleted = storage.delete(doc_key)
    assert deleted is True
    assert storage.exists(doc_key) is False

    # Retrieve non-existing raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        storage.retrieve(doc_key)


def test_local_document_storage_path_traversal_prevention(tmp_path: Path) -> None:
    storage_dir = tmp_path / "test_docs"
    storage = LocalDocumentStorage(root_dir=storage_dir)

    # Attempt path traversal
    storage.store("../../evil.txt", b"evil")
    assert not (tmp_path / "evil.txt").exists()
    assert (storage_dir / "evil.txt").exists()

"""Integration fixtures: temporary PostgreSQL with pgvector and real migrations."""

import logging
import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import psycopg
import pytest

from app.core.config import get_settings
from app.core.database import DatabaseSessionManager

ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_SECRET = "stage4-synthetic-test-secret-with-at-least-32-characters"


def _run_migrations(uri: str) -> None:
    """Apply the real Alembic chain so the vector column and indexes are exercised."""

    from alembic import command
    from alembic.config import Config

    previous_database_url = os.environ.get("DATABASE_URL")
    previous_secret = os.environ.get("APP_SECRET_KEY")
    os.environ["DATABASE_URL"] = uri.replace("postgresql://", "postgresql+psycopg://", 1)
    os.environ["APP_SECRET_KEY"] = SYNTHETIC_SECRET
    get_settings.cache_clear()
    try:
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "alembic"))
        command.upgrade(config, "head")
    finally:
        for name, previous in (
            ("DATABASE_URL", previous_database_url),
            ("APP_SECRET_KEY", previous_secret),
        ):
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        get_settings.cache_clear()


@pytest.fixture(scope="session")
def postgres_uri(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Start an embedded PostgreSQL 16 + pgvector instance for the test session."""

    pgserver = pytest.importorskip(
        "pgserver", reason="pgserver provides embedded PostgreSQL with pgvector"
    )
    # Its atexit handler writes to a closed stream after pytest finishes.
    logging.getLogger("pgserver").disabled = True
    logging.raiseExceptions = False
    data_directory = tmp_path_factory.mktemp("pgdata")
    server = pgserver.get_server(str(data_directory))
    uri = server.get_uri()
    with psycopg.connect(uri, autocommit=True) as connection:
        connection.execute("create extension if not exists vector")
    _run_migrations(uri)
    try:
        yield uri
    finally:
        server.cleanup()


@pytest.fixture
def postgres_connection(postgres_uri: str) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    """Give each test a clean synthetic patient/document/vector state."""

    with psycopg.connect(postgres_uri, autocommit=True) as connection:
        connection.execute("truncate table chunk_embeddings, documents, patient_grants")
        yield connection


@pytest.fixture
async def sessions(
    postgres_uri: str, postgres_connection: psycopg.Connection[tuple[object, ...]]
) -> AsyncIterator[DatabaseSessionManager]:
    manager = DatabaseSessionManager(
        postgres_uri.replace("postgresql://", "postgresql+psycopg://", 1)
    )
    try:
        yield manager
    finally:
        await manager.close()

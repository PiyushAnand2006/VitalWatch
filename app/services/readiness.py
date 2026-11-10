"""Readiness checks for configuration, private storage, database, and embeddings."""

import asyncio
import os
from pathlib import Path
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings
from app.schemas.embeddings import EmbeddingModelInfo
from app.schemas.health import ReadinessCheck

_DATABASE_TIMEOUT_SECONDS = 2.0


class ReadinessDatabase(Protocol):
    @property
    def engine(self) -> AsyncEngine: ...


class ReadinessEmbeddings(Protocol):
    @property
    def model_info(self) -> EmbeddingModelInfo | None: ...


class ReadinessService:
    """Evaluate dependencies without exposing host paths, URLs, or secrets."""

    def __init__(
        self,
        settings: Settings,
        database: ReadinessDatabase | None = None,
        embeddings: ReadinessEmbeddings | None = None,
    ) -> None:
        self._settings = settings
        self._database = database
        self._embeddings = embeddings

    async def evaluate(self) -> list[ReadinessCheck]:
        checks = [
            ReadinessCheck(
                name="configuration",
                status="passed",
                message="Configuration is valid.",
            ),
            self._check_directory(
                "document_storage",
                self._settings.document_storage_directory,
                "Document storage",
            ),
        ]
        if self._database is not None:
            checks.append(await self._check_database())
        if self._embeddings is not None:
            checks.append(self._check_embeddings())
        return checks

    @staticmethod
    def _check_directory(name: str, path: Path, label: str) -> ReadinessCheck:
        try:
            available = path.is_dir() and os.access(path, os.R_OK | os.W_OK | os.X_OK)
        except OSError:
            available = False
        return ReadinessCheck(
            name=name,
            status="passed" if available else "failed",
            message=f"{label} is {'available' if available else 'unavailable'}.",
        )

    async def _check_database(self) -> ReadinessCheck:
        if self._database is None:
            return ReadinessCheck(
                name="database", status="skipped", message="Database is not configured."
            )
        try:
            async with asyncio.timeout(_DATABASE_TIMEOUT_SECONDS):
                async with self._database.engine.connect() as connection:
                    await connection.execute(text("SELECT 1"))
        except Exception:
            return ReadinessCheck(
                name="database",
                status="failed",
                message="Database connection is unavailable.",
            )
        return ReadinessCheck(
            name="database", status="passed", message="Database connection is available."
        )

    def _check_embeddings(self) -> ReadinessCheck:
        info = self._embeddings.model_info if self._embeddings is not None else None
        if info is None:
            return ReadinessCheck(
                name="embedding_model",
                status="skipped",
                message="Embedding model has not been loaded yet; it loads on first use.",
            )
        return ReadinessCheck(
            name="embedding_model",
            status="passed",
            message=f"Embedding model {info.name} is loaded ({info.dimension} dimensions).",
        )

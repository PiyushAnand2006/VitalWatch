"""Readiness checks for dependencies present in the current implementation stage."""

import os
from pathlib import Path

from app.core.config import Settings
from app.schemas.health import ReadinessCheck


class ReadinessService:
    """Evaluate configuration and private storage without exposing host paths."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def evaluate(self) -> list[ReadinessCheck]:
        return [
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

"""Typed environment configuration with secure defaults."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.embeddings import DEFAULT_EMBEDDING_MODEL_NAME


class AppEnvironment(StrEnum):
    """Supported deployment environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


_FORBIDDEN_SECRETS = {
    "replace_me",
    "change_me",
    "changeme",
    "replace_with_secure_random_secret_at_least_32_chars",
}


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "VitalWatch Part 2"
    app_version: str = "0.1.0"
    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    app_secret_key: SecretStr = Field(min_length=32)
    debug: bool = False
    api_docs_enabled: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/vitalwatch"
    document_storage_directory: Path = Path("./data/documents")
    document_storage_encryption_key: SecretStr | None = None
    upload_max_mb: int = Field(default=20, ge=1, le=100)
    upload_max_pages: int = Field(default=200, ge=1, le=1000)
    ocr_language: str = Field(default="eng", min_length=2, max_length=16)
    tesseract_command: Path | None = None

    embedding_model_name: str = Field(
        default=DEFAULT_EMBEDDING_MODEL_NAME, min_length=1, max_length=255
    )
    embedding_model_revision: str = Field(default="main", min_length=1, max_length=128)
    embedding_model_directory: Path = Path("./data/models")
    embedding_device: Literal["cpu", "cuda", "mps", "auto"] = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    embedding_local_files_only: bool = False

    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_min_score: float = Field(default=0.25, ge=0.0, le=1.0)
    retrieval_max_query_chars: int = Field(default=2000, ge=1, le=8000)

    privacy_gate_enabled: bool = True
    privacy_local_only: bool = True
    privacy_approval_ttl_seconds: int = Field(default=300, ge=30, le=3600)
    privacy_max_question_chars: int = Field(default=2000, ge=1, le=8000)
    privacy_nlp_model: str = Field(default="en_core_web_sm", min_length=1, max_length=128)
    privacy_analyzer_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    external_llm_enabled: bool = False

    @field_validator("app_secret_key")
    @classmethod
    def reject_placeholder_secrets(cls, value: SecretStr) -> SecretStr:
        """Reject known placeholders before the application can start."""

        if value.get_secret_value().strip().lower() in _FORBIDDEN_SECRETS:
            raise ValueError("APP_SECRET_KEY must be replaced with a secure random value")
        return value

    @model_validator(mode="after")
    def enforce_security_boundaries(self) -> "Settings":
        """Enforce controls that are not allowed to be disabled."""

        if not self.privacy_gate_enabled:
            raise ValueError("PRIVACY_GATE_ENABLED cannot be false")
        if not self.privacy_local_only and not self.external_llm_enabled:
            raise ValueError(
                "PRIVACY_LOCAL_ONLY cannot be false while EXTERNAL_LLM_ENABLED is false"
            )
        if self.app_env == AppEnvironment.PRODUCTION and self.debug:
            raise ValueError("DEBUG cannot be enabled in production")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache process settings after all validation succeeds."""

    return Settings()

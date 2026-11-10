"""Typed environment configuration with secure defaults."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.endpoints import validate_remote_endpoint
from app.schemas.embeddings import DEFAULT_EMBEDDING_MODEL_NAME
from app.schemas.orchestration import UrgentRule
from app.schemas.providers import ProviderKind


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

    privacy_local_only: bool = True
    privacy_approval_ttl_seconds: int = Field(default=300, ge=30, le=3600)
    privacy_max_question_chars: int = Field(default=2000, ge=1, le=8000)
    privacy_nlp_model: str = Field(default="en_core_web_sm", min_length=1, max_length=128)
    privacy_analyzer_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    external_llm_enabled: bool = False

    nvidia_api_key: SecretStr | None = None
    nvidia_chat_endpoint: str = "https://integrate.api.nvidia.com/v1/chat/completions"
    google_api_key: SecretStr | None = None
    google_api_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    llama_api_key: SecretStr | None = None
    llama_base_url: str | None = None
    llama_chat_route: str = "/v1/chat/completions"
    llama_auth_header: str = "Authorization"
    llama_auth_scheme: str = "Bearer"

    llm_reasoner_a_provider: ProviderKind | None = None
    llm_reasoner_a_model: str | None = Field(default=None, max_length=128)
    llm_reasoner_a_endpoint: str | None = None
    llm_reasoner_b_provider: ProviderKind | None = None
    llm_reasoner_b_model: str | None = Field(default=None, max_length=128)
    llm_reasoner_b_endpoint: str | None = None
    llm_reasoner_c_provider: ProviderKind | None = None
    llm_reasoner_c_model: str | None = Field(default=None, max_length=128)
    llm_reasoner_c_endpoint: str | None = None
    llm_adjudicator_provider: ProviderKind | None = None
    llm_adjudicator_model: str | None = Field(default=None, max_length=128)
    llm_adjudicator_endpoint: str | None = None
    llm_fallback_provider: ProviderKind | None = None
    llm_fallback_model: str | None = Field(default=None, max_length=128)
    llm_fallback_endpoint: str | None = None
    llm_fallback_enabled: bool = False

    llm_request_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_retry_base_delay_seconds: float = Field(default=0.5, ge=0.0, le=10.0)
    llm_rate_limit_per_minute: int = Field(default=30, ge=1, le=600)
    llm_max_output_tokens: int = Field(default=2048, ge=1, le=8192)
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    orchestration_max_debate_rounds: int = Field(default=1, ge=1, le=2)
    orchestration_max_provider_calls: int = Field(default=12, ge=3, le=40)
    orchestration_max_total_tokens: int = Field(default=40000, ge=1000, le=200000)
    orchestration_timeout_seconds: float = Field(default=120.0, ge=10.0, le=600.0)
    safety_urgent_rules: list[UrgentRule] = Field(default_factory=list)

    @field_validator("app_secret_key")
    @classmethod
    def reject_placeholder_secrets(cls, value: SecretStr) -> SecretStr:
        """Reject known placeholders before the application can start."""

        if value.get_secret_value().strip().lower() in _FORBIDDEN_SECRETS:
            raise ValueError("APP_SECRET_KEY must be replaced with a secure random value")
        return value

    @field_validator(
        "nvidia_api_key",
        "google_api_key",
        "llama_api_key",
        "llama_base_url",
        "llm_reasoner_a_provider",
        "llm_reasoner_a_model",
        "llm_reasoner_a_endpoint",
        "llm_reasoner_b_provider",
        "llm_reasoner_b_model",
        "llm_reasoner_b_endpoint",
        "llm_reasoner_c_provider",
        "llm_reasoner_c_model",
        "llm_reasoner_c_endpoint",
        "llm_adjudicator_provider",
        "llm_adjudicator_model",
        "llm_adjudicator_endpoint",
        "llm_fallback_provider",
        "llm_fallback_model",
        "llm_fallback_endpoint",
        mode="before",
    )
    @classmethod
    def blank_optional_values_are_unset(cls, value: object) -> object:
        """Treat empty environment placeholders as unconfigured rather than valid."""

        return None if value == "" else value

    @field_validator(
        "nvidia_chat_endpoint",
        "google_api_base_url",
        "llama_base_url",
        "llm_reasoner_a_endpoint",
        "llm_reasoner_b_endpoint",
        "llm_reasoner_c_endpoint",
        "llm_adjudicator_endpoint",
        "llm_fallback_endpoint",
    )
    @classmethod
    def require_tls_for_remote_endpoints(cls, value: str | None) -> str | None:
        """Reject credentials, fragments, and plaintext non-loopback destinations."""

        return validate_remote_endpoint(value)

    @field_validator("llama_chat_route")
    @classmethod
    def validate_llama_route(cls, value: str) -> str:
        """Keep the configured route an absolute path without query parameters."""

        if not value.startswith("/") or value.startswith("//"):
            raise ValueError("LLAMA_CHAT_ROUTE must be an absolute path")
        if any(character in value for character in "?#") or any(
            character.isspace() for character in value
        ):
            raise ValueError("LLAMA_CHAT_ROUTE must not contain a query, fragment, or whitespace")
        return value

    @field_validator("llama_auth_header")
    @classmethod
    def validate_llama_auth_header(cls, value: str) -> str:
        """Allow a standard header name or an empty value to disable authentication."""

        if value and not all(character.isalnum() or character == "-" for character in value):
            raise ValueError("LLAMA_AUTH_HEADER must be a valid HTTP header name")
        return value

    @field_validator("llama_auth_scheme")
    @classmethod
    def validate_llama_auth_scheme(cls, value: str) -> str:
        """Allow a conventional scheme token or an empty value for raw keys."""

        if value and not all(character.isalnum() or character in "-._" for character in value):
            raise ValueError("LLAMA_AUTH_SCHEME must be a simple auth scheme token")
        return value

    @model_validator(mode="after")
    def enforce_security_boundaries(self) -> "Settings":
        """Enforce controls that are not allowed to be disabled."""

        if not self.privacy_local_only and not self.external_llm_enabled:
            raise ValueError(
                "PRIVACY_LOCAL_ONLY cannot be false while EXTERNAL_LLM_ENABLED is false"
            )
        if self.app_env == AppEnvironment.PRODUCTION and self.debug:
            raise ValueError("DEBUG cannot be enabled in production")
        self._validate_model_slots()
        return self

    def _validate_model_slots(self) -> None:
        """Require complete, usable slot configuration and an explicit fallback."""

        slots = (
            (
                "reasoner_a",
                self.llm_reasoner_a_provider,
                self.llm_reasoner_a_model,
            ),
            (
                "reasoner_b",
                self.llm_reasoner_b_provider,
                self.llm_reasoner_b_model,
            ),
            (
                "reasoner_c",
                self.llm_reasoner_c_provider,
                self.llm_reasoner_c_model,
            ),
            (
                "adjudicator",
                self.llm_adjudicator_provider,
                self.llm_adjudicator_model,
            ),
            (
                "fallback",
                self.llm_fallback_provider,
                self.llm_fallback_model,
            ),
        )
        for name, provider, model in slots:
            if (provider is None) != (model is None):
                raise ValueError(f"LLM {name} slot requires both a provider and a model")
            if provider is ProviderKind.NVIDIA and self.nvidia_api_key is None:
                raise ValueError(f"LLM {name} slot requires NVIDIA_API_KEY")
            if provider is ProviderKind.GOOGLE and self.google_api_key is None:
                raise ValueError(f"LLM {name} slot requires GOOGLE_API_KEY")
            if provider is ProviderKind.LLAMA and self.llama_base_url is None:
                raise ValueError(f"LLM {name} slot requires LLAMA_BASE_URL")
        if self.llm_fallback_enabled and self.llm_fallback_provider is None:
            raise ValueError(
                "LLM_FALLBACK_ENABLED requires a configured fallback provider and model"
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache process settings after all validation succeeds."""

    return Settings()

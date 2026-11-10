"""Readiness checks with synthetic dependencies."""

from app.core.config import Settings
from app.schemas.embeddings import EmbeddingModelInfo
from app.services.readiness import ReadinessService


class FakeConnection:
    async def __aenter__(self) -> "FakeConnection":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def execute(self, statement: object) -> None:
        return None


class FakeEngine:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    def connect(self) -> FakeConnection:
        if self.failure is not None:
            raise self.failure
        return FakeConnection()


class FakeDatabase:
    def __init__(self, failure: Exception | None = None) -> None:
        self.engine = FakeEngine(failure)


class FakeEmbeddings:
    def __init__(self, model_info: EmbeddingModelInfo | None) -> None:
        self.model_info = model_info


async def test_checks_without_optional_dependencies(settings: Settings) -> None:
    settings.document_storage_directory.mkdir(parents=True)

    checks = await ReadinessService(settings).evaluate()

    assert [check.name for check in checks] == ["configuration", "document_storage"]
    assert all(check.status == "passed" for check in checks)


async def test_database_success_is_reported(settings: Settings) -> None:
    checks = await ReadinessService(settings, FakeDatabase()).evaluate()

    database = next(check for check in checks if check.name == "database")
    assert database.status == "passed"
    assert "postgresql" not in database.message


async def test_database_failure_is_safe_and_blocks_readiness(settings: Settings) -> None:
    checks = await ReadinessService(
        settings, FakeDatabase(RuntimeError("postgresql://secret-host/private"))
    ).evaluate()

    database = next(check for check in checks if check.name == "database")
    assert database.status == "failed"
    assert "secret-host" not in database.message


async def test_unloaded_model_is_reported_as_skipped(settings: Settings) -> None:
    checks = await ReadinessService(settings, embeddings=FakeEmbeddings(None)).evaluate()

    model_check = next(check for check in checks if check.name == "embedding_model")
    assert model_check.status == "skipped"
    assert "first use" in model_check.message


async def test_loaded_model_reports_identity(settings: Settings) -> None:
    info = EmbeddingModelInfo(
        name="sentence-transformers/all-MiniLM-L6-v2",
        revision="synthetic-revision",
        dimension=384,
    )

    checks = await ReadinessService(settings, embeddings=FakeEmbeddings(info)).evaluate()

    model_check = next(check for check in checks if check.name == "embedding_model")
    assert model_check.status == "passed"
    assert info.name in model_check.message
    assert "384" in model_check.message

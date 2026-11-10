"""Synthetic tests for the cached local embedding boundary."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.core.exceptions import AppError
from app.schemas.embeddings import EMBEDDING_DIMENSION
from app.services.embeddings import LocalEmbeddingService, clear_embedding_model_cache


class FakeSentenceTransformer:
    """Deterministic stand-in; constructor calls are counted for cache assertions."""

    instances: list["FakeSentenceTransformer"] = []
    constructor_calls = 0
    dimension = EMBEDDING_DIMENSION
    failure: Exception | None = None

    def __init__(self, model_name: str, **kwargs: object) -> None:
        type(self).constructor_calls += 1
        if type(self).failure is not None:
            raise type(self).failure
        self.model_name = model_name
        self.kwargs = kwargs
        self.encode_calls: list[dict[str, object]] = []
        type(self).instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return type(self).dimension

    def encode(self, texts: Sequence[str], **kwargs: object) -> list[list[float]]:
        self.encode_calls.append({"texts": list(texts), **kwargs})
        return [
            [float((index + position) % 7 + 1) for position in range(EMBEDDING_DIMENSION)]
            for index in range(len(texts))
        ]


@pytest.fixture(autouse=True)
def reset_fake_model() -> None:
    clear_embedding_model_cache()
    FakeSentenceTransformer.instances = []
    FakeSentenceTransformer.constructor_calls = 0
    FakeSentenceTransformer.dimension = EMBEDDING_DIMENSION
    FakeSentenceTransformer.failure = None


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", FakeSentenceTransformer, raising=True
    )
    return FakeSentenceTransformer


def _service(model_name: str = "synthetic/embedding-model") -> LocalEmbeddingService:
    return LocalEmbeddingService(model_name, batch_size=8, device="cpu")


async def test_model_identity_is_recorded_and_dimension_is_384(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    service = _service()

    info = await service.ensure_loaded()

    assert info.dimension == EMBEDDING_DIMENSION == 384
    assert info.name == "synthetic/embedding-model"
    assert info.revision == "main"


async def test_revision_records_resolved_local_snapshot(
    fake_model: type[FakeSentenceTransformer], monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *args, **kwargs: str(Path("/cache/models--synthetic/snapshots") / commit),
        raising=True,
    )

    info = await _service().ensure_loaded()

    assert info.revision == commit


async def test_model_is_loaded_once_and_reused_across_instances(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    await _service().embed_documents(["synthetic chunk one"])
    await _service().embed_documents(["synthetic chunk two"])

    assert fake_model.constructor_calls == 1
    assert len(fake_model.instances) == 1


async def test_clearing_cache_forces_a_fresh_load(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    await _service().embed_documents(["synthetic chunk one"])
    clear_embedding_model_cache()

    await _service().embed_documents(["synthetic chunk two"])

    assert fake_model.constructor_calls == 2


async def test_documents_are_encoded_with_configured_batch_size_and_normalization(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    service = LocalEmbeddingService("synthetic/embedding-model", batch_size=4)

    vectors = await service.embed_documents(["first synthetic chunk", "second synthetic chunk"])

    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
    encoded = fake_model.instances[0].encode_calls[0]
    assert encoded["batch_size"] == 4
    assert encoded["normalize_embeddings"] is True
    assert encoded["texts"] == ["first synthetic chunk", "second synthetic chunk"]


async def test_query_embedding_returns_one_vector(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    vector = await _service().embed_query("synthetic retrieval query")

    assert len(vector) == 384
    assert all(isinstance(component, float) for component in vector)


async def test_empty_document_batch_returns_no_vectors_without_loading(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    service = _service()

    assert await service.embed_documents([]) == []
    assert fake_model.constructor_calls == 0
    assert service.model_info is None


async def test_blank_text_is_rejected(fake_model: type[FakeSentenceTransformer]) -> None:
    with pytest.raises(ValueError, match="blank"):
        await _service().embed_documents(["   "])


async def test_model_load_failure_is_reported_safely(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    fake_model.failure = RuntimeError("synthetic loader crash with /private/model/path")

    with pytest.raises(AppError) as error:
        await _service().ensure_loaded()

    assert error.value.status_code == 503
    assert error.value.code == "embedding_model_unavailable"
    assert "/private/model/path" not in error.value.message


async def test_embedding_fails_safely_when_model_cannot_load(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    fake_model.failure = OSError("missing weights")

    with pytest.raises(AppError) as error:
        await _service().embed_documents(["synthetic chunk"])

    assert error.value.code == "embedding_model_unavailable"
    assert fake_model.constructor_calls == 1


async def test_model_with_wrong_dimension_is_rejected(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    fake_model.dimension = 768

    with pytest.raises(AppError) as error:
        await _service().ensure_loaded()

    assert error.value.status_code == 409
    assert error.value.code == "embedding_model_incompatible"


async def test_wrong_dimension_vectors_are_rejected(
    fake_model: type[FakeSentenceTransformer], monkeypatch: pytest.MonkeyPatch
) -> None:
    class ShortVectorModel(FakeSentenceTransformer):
        def encode(self, texts: Sequence[str], **kwargs: object) -> list[list[float]]:
            return [[0.5] * 12 for _ in texts]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", ShortVectorModel, raising=True)

    with pytest.raises(AppError) as error:
        await _service().embed_documents(["synthetic chunk"])

    assert error.value.code == "embedding_model_incompatible"

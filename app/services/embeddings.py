"""Cached local Sentence Transformer embeddings; no embedding call leaves this host."""

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from anyio import to_thread

from app.core.exceptions import AppError
from app.schemas.embeddings import EMBEDDING_DIMENSION, EmbeddingModelInfo

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingService(Protocol):
    """The only embedding boundary used by indexing and retrieval."""

    async def ensure_loaded(self) -> EmbeddingModelInfo:
        """Load the model if necessary and return its recorded identity."""

        ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed indexable chunks in batches."""

        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed one retrieval query locally."""

        ...


_MODEL_CACHE: dict[
    tuple[str, str, str, str], tuple["SentenceTransformer", str]
] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def clear_embedding_model_cache() -> None:
    """Release cached weights; used by tests and explicit operator reloads."""

    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


class LocalEmbeddingService:
    """Sentence Transformer service with lazy loading, batching, and safe failures."""

    def __init__(
        self,
        model_name: str,
        revision: str = "main",
        cache_directory: Path | None = None,
        device: str = "cpu",
        batch_size: int = 32,
        local_files_only: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._model_name = model_name
        self._revision = revision
        self._cache_directory = cache_directory
        self._device = device
        self._batch_size = batch_size
        self._local_files_only = local_files_only
        self._model: SentenceTransformer | None = None
        self._info: EmbeddingModelInfo | None = None

    @property
    def model_info(self) -> EmbeddingModelInfo | None:
        """Return the recorded identity once the model has been loaded."""

        return self._info

    async def ensure_loaded(self) -> EmbeddingModelInfo:
        """Load weights off the event loop and return model/revision/dimension."""

        if self._info is None:
            await to_thread.run_sync(self._load_model)
        if self._info is None:
            raise AppError(
                503, "embedding_model_unavailable", "The local embedding model is not available."
            )
        return self._info

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunk text locally with deterministic batching."""

        if len(texts) == 0:
            return []
        return await to_thread.run_sync(self._embed, list(texts))

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query locally; callers still enforce patient scope."""

        return (await to_thread.run_sync(self._embed, [text]))[0]

    def _load_model(self) -> None:
        """Load once per configuration and cache the weights across service instances."""

        if self._info is not None:
            return
        key = (
            self._model_name,
            self._revision,
            self._device,
            str(self._cache_directory) if self._cache_directory is not None else "",
        )
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is None:
                model = self._construct_model()
                if model.get_sentence_embedding_dimension() != EMBEDDING_DIMENSION:
                    raise AppError(
                        409,
                        "embedding_model_incompatible",
                        "The configured embedding model does not match the vector index.",
                    )
                cached = (model, self._resolve_revision())
                _MODEL_CACHE[key] = cached
        self._model, revision = cached
        self._info = EmbeddingModelInfo(
            name=self._model_name,
            revision=revision,
            dimension=EMBEDDING_DIMENSION,
        )

    def _construct_model(self) -> "SentenceTransformer":
        """Import and construct the local model, converting every failure to one safe error."""

        try:
            from sentence_transformers import SentenceTransformer

            return SentenceTransformer(
                self._model_name,
                revision=self._revision,
                device=None if self._device == "auto" else self._device,
                cache_folder=(
                    str(self._cache_directory) if self._cache_directory is not None else None
                ),
                local_files_only=self._local_files_only,
            )
        except AppError:
            raise
        except Exception as error:
            raise AppError(
                503,
                "embedding_model_unavailable",
                "The local embedding model could not be loaded.",
            ) from error

    def _resolve_revision(self) -> str:
        """Record the resolved Hugging Face snapshot commit when one is cached locally."""

        if Path(self._model_name).expanduser().is_dir():
            return self._revision
        try:
            from huggingface_hub import snapshot_download

            snapshot = snapshot_download(
                self._model_name,
                revision=self._revision,
                cache_dir=(
                    str(self._cache_directory) if self._cache_directory is not None else None
                ),
                local_files_only=True,
            )
        except Exception:
            return self._revision
        resolved = Path(snapshot).name
        return resolved or self._revision

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        """CPU-bound encode step; always executed in a worker thread."""

        if any(not text.strip() for text in texts):
            raise ValueError("embedding text must not be blank")
        if self._model is None:
            self._load_model()
        model = self._model
        if model is None:
            raise AppError(
                503, "embedding_model_unavailable", "The local embedding model is not available."
            )
        encoded = model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = [[float(component) for component in row] for row in encoded]
        if any(len(vector) != EMBEDDING_DIMENSION for vector in vectors):
            raise AppError(
                409,
                "embedding_model_incompatible",
                "The configured embedding model does not match the vector index.",
            )
        if len(vectors) != len(texts):
            raise AppError(
                503, "embedding_model_unavailable", "The local embedding model returned no vector."
            )
        return vectors

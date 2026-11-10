"""Connect private extraction output to local embeddings and pgvector indexing."""

import json
from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from pydantic import ValidationError

from app.core.exceptions import AppError
from app.schemas.embeddings import EmbeddedChunk, EmbeddingModelInfo, IndexResult
from app.schemas.extraction import DocumentChunk
from app.services.embeddings import EmbeddingService


class IndexingRepository(Protocol):
    """Server-controlled vector writes; the caller never supplies SQL."""

    async def replace_document(
        self,
        *,
        patient_id: str,
        document_id: UUID,
        chunks: Sequence[EmbeddedChunk],
        model: EmbeddingModelInfo,
    ) -> int: ...

    async def delete_document(self, patient_id: str, document_id: UUID) -> int: ...


class IndexingStorage(Protocol):
    def retrieve(self, key: str) -> bytes: ...


class DocumentIndexer:
    """Re-index one document from its stored extraction output, atomically per document."""

    def __init__(
        self,
        repository: IndexingRepository,
        storage: IndexingStorage,
        embeddings: EmbeddingService,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._embeddings = embeddings

    async def index_document(self, patient_id: str, document_id: UUID) -> IndexResult:
        chunks = self._read_chunks(document_id)
        model = await self._embeddings.ensure_loaded()
        vectors = await self._embeddings.embed_documents([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise AppError(
                503, "embedding_model_unavailable", "The local embedding model returned no vector."
            )
        embedded = [
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                content_hash=chunk.content_hash,
                page_number=chunk.page_number,
                section=chunk.section,
                text=chunk.text,
                method=chunk.method.value,
                quality=chunk.quality,
                embedding=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        indexed = await self._repository.replace_document(
            patient_id=patient_id,
            document_id=document_id,
            chunks=embedded,
            model=model,
        )
        return IndexResult(document_id=document_id, chunks_indexed=indexed, model=model)

    async def delete_document(self, patient_id: str, document_id: UUID) -> int:
        return await self._repository.delete_document(patient_id, document_id)

    def _read_chunks(self, document_id: UUID) -> list[DocumentChunk]:
        """Read only private extraction output; never re-read the original document."""

        try:
            payload = json.loads(self._storage.retrieve(f"{document_id}.extraction.json"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AppError(
                409,
                "document_not_indexable",
                "The document has no valid extracted text to index.",
            ) from error
        try:
            return [DocumentChunk.model_validate(item) for item in payload["chunks"]]
        except (KeyError, TypeError, ValidationError) as error:
            raise AppError(
                409,
                "document_not_indexable",
                "The document has no valid extracted text to index.",
            ) from error


"""Controlled cross-document rebuild of the local vector index."""

from collections.abc import Sequence
from typing import Protocol
from uuid import UUID

from app.core.exceptions import AppError
from app.schemas.embeddings import EmbeddingModelInfo, IndexRebuildReport, IndexResult
from app.services.embeddings import EmbeddingService


class IndexedDocumentSource(Protocol):
    async def list_processed(
        self, limit: int = 100, after: UUID | None = None
    ) -> Sequence[tuple[str, UUID]]: ...


class IndexStore(Protocol):
    async def delete_index(self) -> int: ...


class IndexingService(Protocol):
    async def index_document(self, patient_id: str, document_id: UUID) -> IndexResult: ...


class IndexRebuildService:
    """Clear and rebuild the index for every processed document, model-identity safe."""

    def __init__(
        self,
        documents: IndexedDocumentSource,
        store: IndexStore,
        indexer: IndexingService,
        embeddings: EmbeddingService,
    ) -> None:
        self._documents = documents
        self._store = store
        self._indexer = indexer
        self._embeddings = embeddings

    async def rebuild(self, batch_size: int = 50) -> IndexRebuildReport:
        """Destructive but explicit: the caller must confirm before invoking this."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        model: EmbeddingModelInfo = await self._embeddings.ensure_loaded()
        cleared = await self._store.delete_index()
        indexed_documents = 0
        chunks_indexed = 0
        failed: list[UUID] = []
        after: UUID | None = None
        while True:
            batch = await self._documents.list_processed(limit=batch_size, after=after)
            if not batch:
                break
            for patient_id, document_id in batch:
                try:
                    result = await self._indexer.index_document(patient_id, document_id)
                except AppError:
                    failed.append(document_id)
                else:
                    indexed_documents += 1
                    chunks_indexed += result.chunks_indexed
                after = document_id
        return IndexRebuildReport(
            model=model,
            cleared_vectors=cleared,
            indexed_documents=indexed_documents,
            chunks_indexed=chunks_indexed,
            failed_document_ids=failed,
        )

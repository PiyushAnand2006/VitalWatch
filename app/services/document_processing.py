"""Connect private originals to Stage 3 extraction without blocking requests."""

import json
from typing import Protocol
from uuid import UUID

from app.schemas.documents import DocumentStatus, DocumentSummary
from app.services.extraction import DocumentExtractor, chunk_document


class ProcessingRepository(Protocol):
    async def get(
        self, patient_id: str, document_id: UUID
    ) -> tuple[DocumentSummary, str] | None: ...

    async def update_status(
        self, patient_id: str, document_id: UUID, status: DocumentStatus
    ) -> DocumentSummary | None: ...


class ProcessingStorage(Protocol):
    def retrieve(self, key: str) -> bytes: ...

    def store(self, key: str, data: bytes) -> str: ...


class ProcessingIndexer(Protocol):
    async def index_document(self, patient_id: str, document_id: UUID) -> object: ...


class DocumentProcessor:
    """Process one stored original and retain only private, structured extraction output."""

    def __init__(
        self,
        repository: ProcessingRepository,
        storage: ProcessingStorage,
        extractor: DocumentExtractor,
        indexer: ProcessingIndexer | None = None,
    ) -> None:
        self._repository = repository
        self._storage = storage
        self._extractor = extractor
        self._indexer = indexer

    async def process(self, patient_id: str, document_id: UUID) -> None:
        document = await self._repository.get(patient_id, document_id)
        if document is None:
            return
        summary, storage_key = document
        await self._repository.update_status(patient_id, document_id, DocumentStatus.PROCESSING)
        try:
            extracted = await self._extractor.extract_async(
                summary.source_name, self._storage.retrieve(storage_key)
            )
            chunks = chunk_document(extracted)
            payload = {
                "document_id": str(document_id),
                "pages": [page.model_dump(mode="json") for page in extracted.pages],
                "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
            }
            self._storage.store(
                f"{document_id}.extraction.json", json.dumps(payload).encode("utf-8")
            )
            if self._indexer is not None:
                await self._indexer.index_document(patient_id, document_id)
            await self._repository.update_status(patient_id, document_id, DocumentStatus.PROCESSED)
        except Exception:
            await self._repository.update_status(patient_id, document_id, DocumentStatus.FAILED)

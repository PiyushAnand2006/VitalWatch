"""Integration coverage for the controlled cross-document index rebuild."""

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import UUID

import pytest

from app.core.database import DatabaseSessionManager
from app.models import DocumentModel
from app.repositories.documents import DocumentRepository
from app.repositories.embeddings import ChunkEmbeddingRepository
from app.schemas.extraction import ExtractedDocument, ExtractedPage, ExtractionMethod
from app.services.document_storage import LocalDocumentStorage
from app.services.embeddings import LocalEmbeddingService
from app.services.extraction import chunk_document
from app.services.index_maintenance import IndexRebuildService
from app.services.indexing import DocumentIndexer

pytestmark = pytest.mark.integration

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class _Harness:
    def __init__(self, sessions: DatabaseSessionManager, storage: LocalDocumentStorage) -> None:
        self.sessions = sessions
        self.storage = storage
        self.embeddings = LocalEmbeddingService(MODEL_NAME, device="cpu", batch_size=4)
        self.vectors = ChunkEmbeddingRepository(sessions.sessionmaker)

    async def create(
        self,
        *,
        patient_id: str = "synthetic-patient-a",
        name: str,
        text: str | None,
        status: str = "processed",
    ) -> UUID:
        document = await DocumentRepository(self.sessions.sessionmaker).create(
            DocumentModel(
                patient_id=patient_id,
                uploader_id="synthetic-user",
                source_name=name,
                storage_key=f"{sha256(name.encode()).hexdigest()}.txt",
                media_type="text/plain",
                checksum=sha256((text or name).encode()).hexdigest(),
                size_bytes=len(text or name),
                status=status,
                created_at=datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
            )
        )
        if text is not None:
            extraction = ExtractedDocument(
                pages=[
                    ExtractedPage(
                        page_number=1,
                        text=text,
                        method=ExtractionMethod.TEXT,
                        quality="high",
                    )
                ]
            )
            self.storage.store(
                f"{document.id}.extraction.json",
                json.dumps(
                    {
                        "document_id": str(document.id),
                        "pages": [
                            page.model_dump(mode="json") for page in extraction.pages
                        ],
                        "chunks": [
                            chunk.model_dump(mode="json")
                            for chunk in chunk_document(extraction)
                        ],
                    }
                ).encode("utf-8"),
            )
        return document.id


async def test_rebuild_recreates_the_whole_index_and_reports_failures(
    sessions: DatabaseSessionManager, tmp_path: Path
) -> None:
    harness = _Harness(sessions, LocalDocumentStorage(tmp_path / "documents"))
    first = await harness.create(name="first.txt", text="Sodium 140 mmol/L\n")
    second = await harness.create(name="second.txt", text="Potassium 4.10 mmol/L\n")
    broken = await harness.create(name="broken.txt", text=None)
    indexer = DocumentIndexer(harness.vectors, harness.storage, harness.embeddings)
    assert (await indexer.index_document("synthetic-patient-a", first)).chunks_indexed == 1
    assert (await indexer.index_document("synthetic-patient-a", second)).chunks_indexed == 1
    before = await harness.vectors.count("synthetic-patient-a")

    report = await IndexRebuildService(
        DocumentRepository(sessions.sessionmaker),
        harness.vectors,
        indexer,
        harness.embeddings,
    ).rebuild(batch_size=10)

    assert report.cleared_vectors == before
    assert report.indexed_documents == 2
    assert report.chunks_indexed == 2
    assert report.failed_document_ids == [broken]
    assert report.model.name == MODEL_NAME
    assert await harness.vectors.count("synthetic-patient-a", first) == 1
    assert await harness.vectors.count("synthetic-patient-a", second) == 1
    assert await harness.vectors.count("synthetic-patient-a", broken) == 0


async def test_rebuild_with_no_processed_documents_clears_the_index(
    sessions: DatabaseSessionManager, tmp_path: Path
) -> None:
    harness = _Harness(sessions, LocalDocumentStorage(tmp_path / "documents"))
    await harness.create(name="uploaded.txt", text="Sodium 140 mmol/L\n", status="uploaded")

    report = await IndexRebuildService(
        DocumentRepository(sessions.sessionmaker),
        harness.vectors,
        DocumentIndexer(harness.vectors, harness.storage, harness.embeddings),
        harness.embeddings,
    ).rebuild()

    assert report.cleared_vectors == 0
    assert report.indexed_documents == 0
    assert report.failed_document_ids == []

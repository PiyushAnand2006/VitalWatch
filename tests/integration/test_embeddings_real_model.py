"""Gate tests using the actual local all-MiniLM-L6-v2 model and real pgvector."""

import json
import math
from hashlib import sha256
from pathlib import Path

import pytest

from app.core.database import DatabaseSessionManager
from app.models import DocumentModel
from app.repositories.documents import DocumentRepository
from app.repositories.embeddings import ChunkEmbeddingRepository
from app.schemas.embeddings import DEFAULT_EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION
from app.schemas.extraction import ExtractedDocument, ExtractedPage, ExtractionMethod
from app.services.document_storage import LocalDocumentStorage
from app.services.embeddings import LocalEmbeddingService, clear_embedding_model_cache
from app.services.extraction import chunk_document
from app.services.indexing import DocumentIndexer

pytestmark = pytest.mark.integration


def _service() -> LocalEmbeddingService:
    return LocalEmbeddingService(
        DEFAULT_EMBEDDING_MODEL_NAME,
        device="cpu",
        batch_size=4,
        local_files_only=False,
    )


def _cosine(left: list[float], right: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


@pytest.fixture(scope="module")
def embeddings() -> LocalEmbeddingService:
    return _service()


async def test_real_model_reports_identity_and_384_dimensions(
    embeddings: LocalEmbeddingService,
) -> None:
    info = await embeddings.ensure_loaded()
    vector = await embeddings.embed_query("synthetic progress note")

    assert info.name == DEFAULT_EMBEDDING_MODEL_NAME
    assert info.dimension == EMBEDDING_DIMENSION == 384
    assert info.revision
    assert len(vector) == 384


async def test_real_embeddings_are_normalized_and_deterministic(
    embeddings: LocalEmbeddingService,
) -> None:
    first = await embeddings.embed_query("synthetic note with a stable sentence")
    second = await embeddings.embed_query("synthetic note with a stable sentence")
    batched = await embeddings.embed_documents(
        ["synthetic note with a stable sentence", "unrelated synthetic sentence"]
    )

    norm = math.sqrt(math.fsum(component * component for component in first))
    assert norm == pytest.approx(1.0, abs=1e-5)
    assert first == pytest.approx(second, abs=1e-7)
    assert first == pytest.approx(batched[0], abs=1e-5)
    assert first != pytest.approx(batched[1], abs=1e-3)


async def test_real_embeddings_rank_related_notes_closer(
    embeddings: LocalEmbeddingService,
) -> None:
    anchor = await embeddings.embed_query("the patient has a high fever and chills")
    related = await embeddings.embed_query("temperature is elevated with shivering")
    unrelated = await embeddings.embed_query("serum sodium is 140 mmol/L")

    assert _cosine(anchor, related) > _cosine(anchor, unrelated)


async def test_weights_are_reloaded_consistently_after_cache_clear(
    embeddings: LocalEmbeddingService,
) -> None:
    before = await embeddings.embed_query("synthetic persistent reload sentence")
    clear_embedding_model_cache()
    reloaded_service = _service()
    after = await reloaded_service.embed_query("synthetic persistent reload sentence")
    info = await reloaded_service.ensure_loaded()

    assert after == pytest.approx(before, abs=1e-6)
    assert info.dimension == 384


async def test_processed_document_is_indexed_and_retrieved_from_pgvector(
    sessions: DatabaseSessionManager, tmp_path: Path
) -> None:
    document = await DocumentRepository(sessions.sessionmaker).create(
        DocumentModel(
            patient_id="synthetic-patient-a",
            uploader_id="synthetic-user",
            source_name="synthetic-note.txt",
            storage_key=f"{sha256(b'storage').hexdigest()}.txt",
            media_type="text/plain",
            checksum=sha256(b"synthetic-note").hexdigest(),
            size_bytes=128,
            status="processed",
        )
    )
    storage = LocalDocumentStorage(tmp_path / "documents")
    extraction = ExtractedDocument(
        pages=[
            ExtractedPage(
                page_number=1,
                text="Progress Note:\nPatient reports chest pain and shortness of breath.",
                method=ExtractionMethod.TEXT,
                quality="high",
            ),
            ExtractedPage(
                page_number=2,
                text="Laboratory Results:\nSodium 140 mmol/L\nPotassium 4.1 mmol/L",
                method=ExtractionMethod.TEXT,
                quality="high",
            ),
        ]
    )
    storage.store(
        f"{document.id}.extraction.json",
        json.dumps(
            {
                "document_id": str(document.id),
                "pages": [page.model_dump(mode="json") for page in extraction.pages],
                "chunks": [
                    chunk.model_dump(mode="json") for chunk in chunk_document(extraction)
                ],
            }
        ).encode("utf-8"),
    )
    vectors = ChunkEmbeddingRepository(sessions.sessionmaker)
    indexer = DocumentIndexer(vectors, storage, _service())

    result = await indexer.index_document("synthetic-patient-a", document.id)

    assert result.chunks_indexed == 2
    assert result.model.dimension == 384
    query = await _service().embed_query("chest pain and trouble breathing")
    matches = await vectors.search(
        patient_id="synthetic-patient-a",
        query_embedding=query,
        model=result.model,
        top_k=2,
    )
    assert len(matches) == 2
    assert "chest pain" in matches[0].text
    assert matches[0].page_number == 1
    assert matches[0].score > matches[1].score


async def test_real_embeddings_cannot_be_queried_for_another_patient(
    sessions: DatabaseSessionManager, tmp_path: Path
) -> None:
    documents = DocumentRepository(sessions.sessionmaker)
    vectors = ChunkEmbeddingRepository(sessions.sessionmaker)
    document = await documents.create(
        DocumentModel(
            patient_id="synthetic-patient-a",
            uploader_id="synthetic-user",
            source_name="synthetic-note.txt",
            storage_key=f"{sha256(b'isolated-storage').hexdigest()}.txt",
            media_type="text/plain",
            checksum=sha256(b"isolated-note").hexdigest(),
            size_bytes=128,
            status="processed",
        )
    )
    storage = LocalDocumentStorage(tmp_path / "documents")
    extraction = ExtractedDocument(
        pages=[
            ExtractedPage(
                page_number=1,
                text="Progress Note:\nSynthetic isolated patient note.",
                method=ExtractionMethod.TEXT,
                quality="high",
            )
        ]
    )
    storage.store(
        f"{document.id}.extraction.json",
        json.dumps(
            {"chunks": [chunk.model_dump(mode="json") for chunk in chunk_document(extraction)]}
        ).encode("utf-8"),
    )
    indexer = DocumentIndexer(vectors, storage, _service())
    result = await indexer.index_document("synthetic-patient-a", document.id)
    query = await _service().embed_query("isolated patient note")

    foreign = await vectors.search(
        patient_id="synthetic-patient-b",
        query_embedding=query,
        model=result.model,
    )

    assert foreign == []


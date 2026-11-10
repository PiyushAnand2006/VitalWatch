"""Synthetic tests for connecting extraction output to local vector indexing."""

import json
from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import AppError
from app.schemas.embeddings import EmbeddedChunk, EmbeddingModelInfo, IndexResult
from app.schemas.extraction import ExtractedDocument, ExtractedPage, ExtractionMethod
from app.services.extraction import chunk_document
from app.services.indexing import DocumentIndexer

MODEL = EmbeddingModelInfo(
    name="sentence-transformers/all-MiniLM-L6-v2", revision="synthetic-revision", dimension=384
)


class FakeStorage:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def retrieve(self, key: str) -> bytes:
        if key not in self.files:
            raise FileNotFoundError(key)
        return self.files[key]


class FakeEmbeddings:
    def __init__(self) -> None:
        self.embedded_texts: list[str] = []
        self.loaded = False

    async def ensure_loaded(self) -> EmbeddingModelInfo:
        self.loaded = True
        return MODEL

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded_texts = list(texts)
        return [[0.01 * (index + 1)] * 384 for index in range(len(texts))]

    async def embed_query(self, text: str) -> list[float]:
        return [0.5] * 384


class FakeRepository:
    def __init__(self) -> None:
        self.replaced: list[tuple[str, UUID, list[EmbeddedChunk], EmbeddingModelInfo]] = []
        self.deleted: list[tuple[str, UUID]] = []

    async def replace_document(
        self,
        *,
        patient_id: str,
        document_id: UUID,
        chunks: Sequence[EmbeddedChunk],
        model: EmbeddingModelInfo,
    ) -> int:
        self.replaced.append((patient_id, document_id, list(chunks), model))
        return len(chunks)

    async def delete_document(self, patient_id: str, document_id: UUID) -> int:
        self.deleted.append((patient_id, document_id))
        return 3


def _extraction(document_id: UUID) -> bytes:
    chunks = chunk_document(
        ExtractedDocument(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text="Laboratory Results:\nSodium 140 mmol/L\nHeart rate 118 bpm",
                    method=ExtractionMethod.TEXT,
                    quality="high",
                )
            ]
        )
    )
    return json.dumps(
        {
            "document_id": str(document_id),
            "pages": [],
            "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
        }
    ).encode("utf-8")


async def test_index_document_preserves_provenance_and_model_identity() -> None:
    document_id = uuid4()
    storage = FakeStorage({f"{document_id}.extraction.json": _extraction(document_id)})
    embeddings = FakeEmbeddings()
    repository = FakeRepository()

    result = await DocumentIndexer(repository, storage, embeddings).index_document(
        "synthetic-patient", document_id
    )

    assert isinstance(result, IndexResult)
    assert result.document_id == document_id
    assert result.chunks_indexed == 1
    assert result.model == MODEL
    assert embeddings.loaded is True
    assert embeddings.embedded_texts == [
        "Laboratory Results:\nSodium 140 mmol/L\nHeart rate 118 bpm"
    ]
    patient_id, stored_id, chunks, model = repository.replaced[0]
    assert (patient_id, stored_id, model) == ("synthetic-patient", document_id, MODEL)
    assert chunks[0].chunk_id == "p1-c1"
    assert chunks[0].page_number == 1
    assert chunks[0].method == "text"
    assert len(chunks[0].embedding) == 384


async def test_missing_extraction_output_is_not_indexable() -> None:
    document_id = uuid4()

    with pytest.raises(AppError) as error:
        await DocumentIndexer(FakeRepository(), FakeStorage({}), FakeEmbeddings()).index_document(
            "synthetic-patient", document_id
        )

    assert error.value.status_code == 409
    assert error.value.code == "document_not_indexable"


async def test_malformed_extraction_output_is_not_indexable() -> None:
    document_id = uuid4()
    storage = FakeStorage({f"{document_id}.extraction.json": b'{"chunks": [{"chunk_id": 1}]}'})

    with pytest.raises(AppError) as error:
        await DocumentIndexer(FakeRepository(), storage, FakeEmbeddings()).index_document(
            "synthetic-patient", document_id
        )

    assert error.value.code == "document_not_indexable"


async def test_document_without_chunks_clears_its_index() -> None:
    document_id = uuid4()
    storage = FakeStorage(
        {
            f"{document_id}.extraction.json": json.dumps(
                {"document_id": str(document_id), "pages": [], "chunks": []}
            ).encode()
        }
    )
    repository = FakeRepository()

    result = await DocumentIndexer(repository, storage, FakeEmbeddings()).index_document(
        "synthetic-patient", document_id
    )

    assert result.chunks_indexed == 0
    assert repository.replaced[0][2] == []




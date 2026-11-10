"""Synthetic tests for the controlled full index rebuild."""

from collections.abc import Sequence
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.cli import main
from app.core.exceptions import AppError
from app.schemas.embeddings import EmbeddingModelInfo, IndexRebuildReport, IndexResult
from app.services.index_maintenance import IndexRebuildService

MODEL = EmbeddingModelInfo(
    name="sentence-transformers/all-MiniLM-L6-v2", revision="synthetic-revision", dimension=384
)


class FakeDocuments:
    def __init__(self, documents: list[tuple[str, UUID]]) -> None:
        self.documents = documents
        self.page_sizes: list[int] = []

    async def list_processed(
        self, limit: int = 100, after: UUID | None = None
    ) -> Sequence[tuple[str, UUID]]:
        self.page_sizes.append(limit)
        eligible = sorted(
            (document for document in self.documents if after is None or document[1] > after),
            key=lambda document: document[1],
        )
        return eligible[:limit]


class FakeStore:
    def __init__(self, cleared: int = 0) -> None:
        self.cleared = cleared
        self.calls = 0

    async def delete_index(self) -> int:
        self.calls += 1
        return self.cleared


class FakeIndexer:
    def __init__(self, failing: set[UUID] | None = None) -> None:
        self.failing = failing or set()
        self.indexed: list[UUID] = []

    async def index_document(self, patient_id: str, document_id: UUID) -> IndexResult:
        if document_id in self.failing:
            raise AppError(409, "document_not_indexable", "Synthetic indexing failure.")
        self.indexed.append(document_id)
        return IndexResult(document_id=document_id, chunks_indexed=2, model=MODEL)


class FakeEmbeddings:
    model_info = MODEL

    async def ensure_loaded(self) -> EmbeddingModelInfo:
        return MODEL

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * 384


def _documents(count: int) -> list[tuple[str, UUID]]:
    return [("synthetic-patient", uuid4()) for _ in range(count)]


async def test_rebuild_clears_and_reindexes_every_processed_document() -> None:
    documents = _documents(3)
    store = FakeStore(cleared=9)
    indexer = FakeIndexer()

    report = await IndexRebuildService(
        FakeDocuments(documents), store, indexer, FakeEmbeddings()
    ).rebuild(batch_size=2)

    assert isinstance(report, IndexRebuildReport)
    assert report.cleared_vectors == 9
    assert report.indexed_documents == 3
    assert report.chunks_indexed == 6
    assert report.failed_document_ids == []
    assert report.model == MODEL
    assert store.calls == 1
    assert sorted(indexer.indexed, key=str) == sorted([doc[1] for doc in documents], key=str)


async def test_rebuild_continues_past_expected_document_failures() -> None:
    documents = _documents(3)
    failing = documents[1][1]
    indexer = FakeIndexer(failing={failing})

    report = await IndexRebuildService(
        FakeDocuments(documents), FakeStore(cleared=3), indexer, FakeEmbeddings()
    ).rebuild(batch_size=10)

    assert report.indexed_documents == 2
    assert report.failed_document_ids == [failing]
    assert report.chunks_indexed == 4


async def test_rebuild_requires_a_positive_batch_size() -> None:
    with pytest.raises(ValueError):
        await IndexRebuildService(
            FakeDocuments([]), FakeStore(), FakeIndexer(), FakeEmbeddings()
        ).rebuild(batch_size=0)


def test_rebuild_cli_refuses_without_confirmation(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["rebuild-index"]) == 2
    assert "--yes" in capsys.readouterr().err


def test_rebuild_cli_requires_a_known_command() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_report_schema_rejects_counts_below_zero() -> None:
    with pytest.raises(ValidationError):
        IndexRebuildReport(
            model=MODEL,
            cleared_vectors=-1,
            indexed_documents=0,
            chunks_indexed=0,
            failed_document_ids=[],
        )


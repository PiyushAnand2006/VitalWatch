"""Synthetic tests for patient-scoped retrieval policy and typed no-evidence states."""

from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID, uuid4

import pytest

from app.core.exceptions import AppError
from app.schemas.embeddings import ChunkMatch, EmbeddingModelInfo
from app.schemas.retrieval import NoEvidenceReason, RetrievalStatus
from app.services.retrieval import RetrievalService

MODEL = EmbeddingModelInfo(
    name="sentence-transformers/all-MiniLM-L6-v2", revision="synthetic-revision", dimension=384
)
CREATED_AT = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)


class FakeEmbeddings:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.model_info = MODEL

    async def ensure_loaded(self) -> EmbeddingModelInfo:
        return MODEL

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1] * 384


class FakeRepository:
    def __init__(self, matches: list[ChunkMatch]) -> None:
        self.matches = matches
        self.calls: list[dict[str, object]] = []

    async def search(
        self,
        *,
        patient_id: str,
        query_embedding: Sequence[float],
        model: EmbeddingModelInfo,
        top_k: int = 5,
        document_ids: Sequence[UUID] | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[ChunkMatch]:
        self.calls.append(
            {
                "patient_id": patient_id,
                "top_k": top_k,
                "document_ids": document_ids,
                "created_from": created_from,
                "created_to": created_to,
                "model": model,
            }
        )
        return self.matches


def _match(
    text: str,
    score: float,
    *,
    chunk_id: str | None = None,
    document_id: UUID | None = None,
    page_number: int = 1,
) -> ChunkMatch:
    resolved_document = document_id or uuid4()
    return ChunkMatch(
        chunk_id=chunk_id or f"p{page_number}-c1",
        document_id=resolved_document,
        patient_id="synthetic-patient-a",
        page_number=page_number,
        section=None,
        text=text,
        content_hash=sha256(text.encode()).hexdigest(),
        method="text",
        quality="high",
        distance=1.0 - score,
        score=score,
        document_created_at=CREATED_AT,
    )


def _service(
    repository: FakeRepository, embeddings: FakeEmbeddings | None = None
) -> RetrievalService:
    return RetrievalService(
        repository, embeddings or FakeEmbeddings(), top_k=3, min_score=0.5, max_query_chars=100
    )


async def test_relevance_threshold_filters_weak_matches() -> None:
    repository = FakeRepository(
        [
            _match("strong match", 0.92, chunk_id="p1-c1"),
            _match("medium match", 0.51, chunk_id="p1-c2"),
            _match("weak match", 0.49, chunk_id="p1-c3"),
        ]
    )

    result = await _service(repository).retrieve(
        patient_id="synthetic-patient-a", query="  synthetic query  "
    )

    assert result.status is RetrievalStatus.RETRIEVED
    assert [item.chunk_id for item in result.items] == ["p1-c1", "p1-c2"]
    assert result.below_threshold == 1
    assert result.candidates_considered == 3
    assert result.query == "synthetic query"
    assert all(item.trust_level == "untrusted_retrieved_text" for item in result.items)


async def test_no_matching_documents_returns_typed_no_evidence() -> None:
    result = await _service(FakeRepository([])).retrieve(
        patient_id="synthetic-patient-a", query="synthetic query"
    )

    assert result.status is RetrievalStatus.NO_EVIDENCE
    assert result.reason is NoEvidenceReason.NO_MATCHING_DOCUMENTS
    assert result.items == []


async def test_all_candidates_below_threshold_report_that_reason() -> None:
    repository = FakeRepository([_match("weak match", 0.2)])

    result = await _service(repository).retrieve(
        patient_id="synthetic-patient-a", query="synthetic query"
    )

    assert result.status is RetrievalStatus.NO_EVIDENCE
    assert result.reason is NoEvidenceReason.BELOW_RELEVANCE_THRESHOLD
    assert result.below_threshold == 1
    assert result.items == []


async def test_overlapping_chunks_are_deduplicated_keeping_the_best_ranked() -> None:
    overlap = "shared synthetic tail " * 8
    document_id = uuid4()
    repository = FakeRepository(
        [
            _match(
                f"first synthetic chunk {overlap}",
                0.95,
                chunk_id="p1-c1",
                document_id=document_id,
            ),
            _match(
                f"second synthetic chunk {overlap}",
                0.90,
                chunk_id="p1-c2",
                document_id=document_id,
            ),
        ]
    )

    result = await _service(repository).retrieve(
        patient_id="synthetic-patient-a", query="synthetic query"
    )

    assert [item.chunk_id for item in result.items] == ["p1-c1"]
    assert result.duplicates_removed == 1


async def test_distinct_chunks_are_not_deduplicated() -> None:
    repository = FakeRepository(
        [
            _match("heart rate is elevated", 0.95, chunk_id="p1-c1"),
            _match("sodium is within range", 0.90, chunk_id="p1-c2"),
        ]
    )

    result = await _service(repository).retrieve(
        patient_id="synthetic-patient-a", query="synthetic query"
    )

    assert len(result.items) == 2
    assert result.duplicates_removed == 0


async def test_identical_text_in_another_document_is_not_deduplicated() -> None:
    repository = FakeRepository(
        [
            _match("identical synthetic note", 0.95, chunk_id="p1-c1"),
            _match("identical synthetic note", 0.90, chunk_id="p1-c1"),
        ]
    )

    result = await _service(repository).retrieve(
        patient_id="synthetic-patient-a", query="synthetic query"
    )

    assert len(result.items) == 2
    assert result.duplicates_removed == 0


async def test_top_k_is_applied_after_threshold_filtering() -> None:
    repository = FakeRepository(
        [
            _match(f"distinct synthetic chunk {index}", 0.9, chunk_id=f"p1-c{index}")
            for index in range(5)
        ]
    )

    result = await _service(repository).retrieve(
        patient_id="synthetic-patient-a", query="synthetic query"
    )

    assert len(result.items) == 3
    assert result.top_k == 3


async def test_filters_are_forwarded_and_as_of_bounds_availability() -> None:
    repository = FakeRepository([])
    document_ids = (uuid4(), uuid4())
    created_from = datetime(2026, 1, 1, tzinfo=UTC)
    created_to = datetime(2026, 6, 1, tzinfo=UTC)
    as_of = datetime(2026, 3, 1, tzinfo=UTC)

    await _service(repository).retrieve(
        patient_id="synthetic-patient-a",
        query="synthetic query",
        as_of=as_of,
        document_ids=document_ids,
        created_from=created_from,
        created_to=created_to,
    )

    call = repository.calls[0]
    assert call["patient_id"] == "synthetic-patient-a"
    assert call["document_ids"] == document_ids
    assert call["created_from"] == created_from
    assert call["created_to"] == as_of
    assert call["top_k"] == 12
    assert call["model"] == MODEL


async def test_empty_document_filter_returns_no_evidence_without_querying() -> None:
    repository = FakeRepository([_match("should not appear", 0.99)])

    result = await _service(repository).retrieve(
        patient_id="synthetic-patient-a", query="synthetic query", document_ids=[]
    )

    assert result.status is RetrievalStatus.NO_EVIDENCE
    assert result.reason is NoEvidenceReason.NO_MATCHING_DOCUMENTS
    assert repository.calls == []


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
async def test_blank_query_is_rejected(query: str) -> None:
    with pytest.raises(AppError) as error:
        await _service(FakeRepository([])).retrieve(
            patient_id="synthetic-patient-a", query=query
        )

    assert error.value.code == "invalid_query"


async def test_oversized_query_is_rejected() -> None:
    with pytest.raises(AppError) as error:
        await _service(FakeRepository([])).retrieve(
            patient_id="synthetic-patient-a", query="x" * 101
        )

    assert error.value.code == "query_too_long"


async def test_naive_as_of_is_rejected() -> None:
    with pytest.raises(AppError) as error:
        await _service(FakeRepository([])).retrieve(
            patient_id="synthetic-patient-a",
            query="synthetic query",
            as_of=datetime(2026, 3, 1),
        )

    assert error.value.code == "invalid_retrieval_filter"


async def test_inverted_date_window_is_rejected() -> None:
    with pytest.raises(AppError) as error:
        await _service(FakeRepository([])).retrieve(
            patient_id="synthetic-patient-a",
            query="synthetic query",
            created_from=datetime(2026, 5, 1, tzinfo=UTC),
            created_to=datetime(2026, 4, 1, tzinfo=UTC),
        )

    assert error.value.code == "invalid_retrieval_filter"

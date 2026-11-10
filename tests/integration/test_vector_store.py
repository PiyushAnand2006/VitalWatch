"""Stage 4 pgvector behaviour against a temporary real PostgreSQL instance."""

from datetime import UTC, datetime
from hashlib import sha256
from math import cos, sin
from uuid import UUID, uuid4

import psycopg
import pytest

from app.core.database import DatabaseSessionManager
from app.core.exceptions import AppError
from app.models import DocumentModel
from app.repositories.documents import DocumentRepository
from app.repositories.embeddings import ChunkEmbeddingRepository
from app.schemas.embeddings import ChunkMatch, EmbeddedChunk, EmbeddingModelInfo

pytestmark = pytest.mark.integration

MODEL = EmbeddingModelInfo(
    name="sentence-transformers/all-MiniLM-L6-v2", revision="synthetic-revision", dimension=384
)

VECTOR_A = [1.0] + [0.0] * 383
VECTOR_B = [0.0, 1.0] + [0.0] * 382
VECTOR_C = [0.0] * 383 + [1.0]


def _document(
    patient_id: str,
    name: str = "synthetic-note.txt",
    created_at: datetime | None = None,
) -> DocumentModel:
    unique = uuid4().hex
    return DocumentModel(
        patient_id=patient_id,
        uploader_id="synthetic-user",
        source_name=name,
        storage_key=f"{unique}.txt",
        media_type="text/plain",
        checksum=sha256(unique.encode()).hexdigest(),
        size_bytes=64,
        status="processed",
        created_at=created_at or datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
    )


async def _store_document(
    sessions: DatabaseSessionManager,
    patient_id: str,
    name: str = "synthetic-note.txt",
    created_at: datetime | None = None,
) -> UUID:
    summary = await DocumentRepository(sessions.sessionmaker).create(
        _document(patient_id, name, created_at)
    )
    return summary.id


def _chunk(chunk_id: str, text: str, vector: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        content_hash=sha256(text.encode()).hexdigest(),
        page_number=1,
        section=None,
        text=text,
        method="text",
        quality="high",
        embedding=vector,
    )


def test_migration_creates_vector_column_and_hnsw_index(
    postgres_connection: psycopg.Connection[tuple[object, ...]],
) -> None:
    column = postgres_connection.execute(
        "select udt_name from information_schema.columns "
        "where table_name = 'chunk_embeddings' and column_name = 'embedding'"
    ).fetchone()
    assert column is not None
    assert column[0] == "vector"

    index_definition = postgres_connection.execute(
        "select indexdef from pg_indexes "
        "where tablename = 'chunk_embeddings' and indexname = 'ix_chunk_embeddings_embedding_hnsw'"
    ).fetchone()
    assert index_definition is not None
    assert "hnsw" in index_definition[0]
    assert "vector_cosine_ops" in index_definition[0]


async def test_vectors_are_stored_and_ranked_by_cosine_distance(
    sessions: DatabaseSessionManager,
) -> None:
    repository = ChunkEmbeddingRepository(sessions.sessionmaker)
    document_id = await _store_document(sessions, "synthetic-patient-a")

    await repository.replace_document(
        patient_id="synthetic-patient-a",
        document_id=document_id,
        chunks=[
            _chunk("p1-c1", "heart rate elevated", VECTOR_A),
            _chunk("p1-c2", "sodium within range", VECTOR_B),
        ],
        model=MODEL,
    )

    matches = await repository.search(
        patient_id="synthetic-patient-a", query_embedding=VECTOR_A, model=MODEL, top_k=2
    )

    assert [match.chunk_id for match in matches] == ["p1-c1", "p1-c2"]
    assert isinstance(matches[0], ChunkMatch)
    assert matches[0].distance == pytest.approx(0.0, abs=1e-6)
    assert matches[0].score == pytest.approx(1.0, abs=1e-6)
    assert matches[1].distance == pytest.approx(1.0, abs=1e-6)
    assert matches[0].document_id == document_id
    assert matches[0].patient_id == "synthetic-patient-a"


async def test_search_never_crosses_patient_boundaries(
    sessions: DatabaseSessionManager,
) -> None:
    repository = ChunkEmbeddingRepository(sessions.sessionmaker)
    document_a = await _store_document(sessions, "synthetic-patient-a")
    document_b = await _store_document(sessions, "synthetic-patient-b")
    await repository.replace_document(
        patient_id="synthetic-patient-a",
        document_id=document_a,
        chunks=[_chunk("p1-c1", "patient a note", VECTOR_A)],
        model=MODEL,
    )
    await repository.replace_document(
        patient_id="synthetic-patient-b",
        document_id=document_b,
        chunks=[_chunk("p1-c1", "patient b note", VECTOR_A)],
        model=MODEL,
    )

    matches_a = await repository.search(
        patient_id="synthetic-patient-a", query_embedding=VECTOR_A, model=MODEL
    )
    matches_b = await repository.search(
        patient_id="synthetic-patient-b", query_embedding=VECTOR_A, model=MODEL
    )

    assert [match.patient_id for match in matches_a] == ["synthetic-patient-a"]
    assert [match.patient_id for match in matches_b] == ["synthetic-patient-b"]
    assert matches_a[0].text == "patient a note"
    assert matches_b[0].text == "patient b note"


async def test_document_filter_and_foreign_document_filter(
    sessions: DatabaseSessionManager,
) -> None:
    repository = ChunkEmbeddingRepository(sessions.sessionmaker)
    document_one = await _store_document(sessions, "synthetic-patient-a", "one.txt")
    document_two = await _store_document(sessions, "synthetic-patient-a", "two.txt")
    foreign_document = await _store_document(sessions, "synthetic-patient-b", "foreign.txt")
    await repository.replace_document(
        patient_id="synthetic-patient-a",
        document_id=document_one,
        chunks=[_chunk("p1-c1", "first document", VECTOR_A)],
        model=MODEL,
    )
    await repository.replace_document(
        patient_id="synthetic-patient-a",
        document_id=document_two,
        chunks=[_chunk("p1-c1", "second document", VECTOR_A)],
        model=MODEL,
    )
    await repository.replace_document(
        patient_id="synthetic-patient-b",
        document_id=foreign_document,
        chunks=[_chunk("p1-c1", "foreign document", VECTOR_A)],
        model=MODEL,
    )

    filtered = await repository.search(
        patient_id="synthetic-patient-a",
        query_embedding=VECTOR_A,
        model=MODEL,
        document_ids=[document_two],
    )
    foreign = await repository.search(
        patient_id="synthetic-patient-a",
        query_embedding=VECTOR_A,
        model=MODEL,
        document_ids=[foreign_document],
    )

    assert [match.document_id for match in filtered] == [document_two]
    assert foreign == []


async def test_reindex_replaces_a_documents_vectors_atomically(
    sessions: DatabaseSessionManager,
) -> None:
    repository = ChunkEmbeddingRepository(sessions.sessionmaker)
    document_id = await _store_document(sessions, "synthetic-patient-a")
    await repository.replace_document(
        patient_id="synthetic-patient-a",
        document_id=document_id,
        chunks=[
            _chunk("p1-c1", "old chunk one", VECTOR_A),
            _chunk("p1-c2", "old chunk two", VECTOR_B),
        ],
        model=MODEL,
    )

    await repository.replace_document(
        patient_id="synthetic-patient-a",
        document_id=document_id,
        chunks=[_chunk("p1-c9", "new chunk", VECTOR_C)],
        model=MODEL,
    )

    matches = await repository.search(
        patient_id="synthetic-patient-a", query_embedding=VECTOR_C, model=MODEL, top_k=10
    )
    assert [match.chunk_id for match in matches] == ["p1-c9"]
    assert await repository.count("synthetic-patient-a", document_id) == 1


async def test_documents_are_scoped_to_their_patient_on_write(
    sessions: DatabaseSessionManager,
) -> None:
    repository = ChunkEmbeddingRepository(sessions.sessionmaker)
    foreign_document = await _store_document(sessions, "synthetic-patient-b")

    with pytest.raises(AppError) as error:
        await repository.replace_document(
            patient_id="synthetic-patient-a",
            document_id=foreign_document,
            chunks=[_chunk("p1-c1", "misattributed chunk", VECTOR_A)],
            model=MODEL,
        )

    assert error.value.status_code == 404
    assert await repository.count("synthetic-patient-b", foreign_document) == 0


async def test_mixing_embedding_models_is_rejected_and_preserves_existing_vectors(
    sessions: DatabaseSessionManager,
) -> None:
    repository = ChunkEmbeddingRepository(sessions.sessionmaker)
    document_id = await _store_document(sessions, "synthetic-patient-a")
    await repository.replace_document(
        patient_id="synthetic-patient-a",
        document_id=document_id,
        chunks=[_chunk("p1-c1", "original chunk", VECTOR_A)],
        model=MODEL,
    )
    other_model = EmbeddingModelInfo(
        name="sentence-transformers/all-MiniLM-L6-v2",
        revision="different-revision",
        dimension=384,
    )

    with pytest.raises(AppError) as error:
        await repository.replace_document(
            patient_id="synthetic-patient-a",
            document_id=document_id,
            chunks=[_chunk("p1-c2", "incompatible chunk", VECTOR_B)],
            model=other_model,
        )

    assert error.value.status_code == 409
    assert error.value.code == "embedding_model_mismatch"
    stored = await repository.search(
        patient_id="synthetic-patient-a", query_embedding=VECTOR_A, model=MODEL
    )
    assert [match.chunk_id for match in stored] == ["p1-c1"]


async def test_search_ignores_chunks_from_an_unknown_model_identity(
    sessions: DatabaseSessionManager,
) -> None:
    repository = ChunkEmbeddingRepository(sessions.sessionmaker)
    document_id = await _store_document(sessions, "synthetic-patient-a")
    await repository.replace_document(
        patient_id="synthetic-patient-a",
        document_id=document_id,
        chunks=[_chunk("p1-c1", "original chunk", VECTOR_A)],
        model=MODEL,
    )
    other_model = EmbeddingModelInfo(name=MODEL.name, revision="unknown", dimension=384)

    matches = await repository.search(
        patient_id="synthetic-patient-a", query_embedding=VECTOR_A, model=other_model
    )

    assert matches == []


async def test_model_dimension_other_than_384_is_rejected(
    sessions: DatabaseSessionManager,
) -> None:
    repository = ChunkEmbeddingRepository(sessions.sessionmaker)
    document_id = await _store_document(sessions, "synthetic-patient-a")
    wrong_model = EmbeddingModelInfo(name=MODEL.name, revision=MODEL.revision, dimension=768)

    with pytest.raises(AppError) as error:
        await repository.replace_document(
            patient_id="synthetic-patient-a",
            document_id=document_id,
            chunks=[],
            model=wrong_model,
        )

    assert error.value.code == "embedding_model_incompatible"


async def test_invalid_query_vector_dimension_is_rejected(
    sessions: DatabaseSessionManager,
) -> None:
    repository = ChunkEmbeddingRepository(sessions.sessionmaker)

    with pytest.raises(AppError) as error:
        await repository.search(
            patient_id="synthetic-patient-a", query_embedding=[1.0, 0.0], model=MODEL
        )

    assert error.value.code == "invalid_query_vector"


async def test_vector_deletion_is_patient_scoped(sessions: DatabaseSessionManager) -> None:
    repository = ChunkEmbeddingRepository(sessions.sessionmaker)
    document_a = await _store_document(sessions, "synthetic-patient-a")
    document_b = await _store_document(sessions, "synthetic-patient-b")
    await repository.replace_document(
        patient_id="synthetic-patient-a",
        document_id=document_a,
        chunks=[_chunk("p1-c1", "patient a", VECTOR_A)],
        model=MODEL,
    )
    await repository.replace_document(
        patient_id="synthetic-patient-b",
        document_id=document_b,
        chunks=[_chunk("p1-c1", "patient b", VECTOR_A)],
        model=MODEL,
    )

    removed = await repository.delete_document("synthetic-patient-b", document_a)

    assert removed == 0
    assert await repository.count("synthetic-patient-a", document_a) == 1
    assert await repository.count("synthetic-patient-b", document_b) == 1


async def test_document_deletion_removes_chunks_and_vectors_in_one_transaction(
    sessions: DatabaseSessionManager,
) -> None:
    documents = DocumentRepository(sessions.sessionmaker)
    vectors = ChunkEmbeddingRepository(sessions.sessionmaker)
    document_a = await documents.create(_document("synthetic-patient-a", "one.txt"))
    document_b = await documents.create(_document("synthetic-patient-a", "two.txt"))
    await vectors.replace_document(
        patient_id="synthetic-patient-a",
        document_id=document_a.id,
        chunks=[_chunk("p1-c1", "first", VECTOR_A)],
        model=MODEL,
    )
    await vectors.replace_document(
        patient_id="synthetic-patient-a",
        document_id=document_b.id,
        chunks=[_chunk("p1-c1", "second", VECTOR_A)],
        model=MODEL,
    )

    storage_key = await documents.delete("synthetic-patient-a", document_a.id)

    assert storage_key is not None
    assert await vectors.count("synthetic-patient-a", document_a.id) == 0
    assert await vectors.count("synthetic-patient-a", document_b.id) == 1
    assert await documents.get("synthetic-patient-a", document_a.id) is None


def _angle_vector(angle: float) -> list[float]:
    return [cos(angle), sin(angle)] + [0.0] * 382


async def test_hnsw_patient_filter_returns_the_exact_cosine_ranking(
    sessions: DatabaseSessionManager,
) -> None:
    repository = ChunkEmbeddingRepository(sessions.sessionmaker)
    angles = {
        "synthetic-patient-a": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        "synthetic-patient-b": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    }
    for patient_id, patient_angles in angles.items():
        document_id = await _store_document(sessions, patient_id)
        await repository.replace_document(
            patient_id=patient_id,
            document_id=document_id,
            chunks=[
                _chunk(f"p1-c{index + 1}", f"chunk {index}", _angle_vector(angle))
                for index, angle in enumerate(patient_angles)
            ],
            model=MODEL,
        )

    for patient_id, patient_angles in angles.items():
        matches = await repository.search(
            patient_id=patient_id, query_embedding=VECTOR_A, model=MODEL, top_k=3
        )
        ranked = sorted(range(len(patient_angles)), key=lambda index: patient_angles[index])

        assert [match.chunk_id for match in matches] == [
            f"p1-c{index + 1}" for index in ranked[:3]
        ]
        assert [match.score for match in matches] == pytest.approx(
            [cos(patient_angles[index]) for index in ranked[:3]], abs=1e-6
        )


async def test_hnsw_date_filter_returns_the_exact_cosine_ranking(
    sessions: DatabaseSessionManager,
) -> None:
    repository = ChunkEmbeddingRepository(sessions.sessionmaker)
    early = await _store_document(
        sessions, "synthetic-patient-a", "early.txt", datetime(2026, 1, 1, tzinfo=UTC)
    )
    late = await _store_document(
        sessions, "synthetic-patient-a", "late.txt", datetime(2026, 6, 1, tzinfo=UTC)
    )
    await repository.replace_document(
        patient_id="synthetic-patient-a",
        document_id=early,
        chunks=[
            _chunk("p1-c1", "early near", _angle_vector(0.05)),
            _chunk("p1-c2", "early far", _angle_vector(0.50)),
        ],
        model=MODEL,
    )
    await repository.replace_document(
        patient_id="synthetic-patient-a",
        document_id=late,
        chunks=[_chunk("p1-c1", "late exact", _angle_vector(0.0))],
        model=MODEL,
    )

    bounded = await repository.search(
        patient_id="synthetic-patient-a",
        query_embedding=VECTOR_A,
        model=MODEL,
        top_k=3,
        created_to=datetime(2026, 2, 1, tzinfo=UTC),
    )
    unbounded = await repository.search(
        patient_id="synthetic-patient-a", query_embedding=VECTOR_A, model=MODEL, top_k=3
    )

    assert [match.document_id for match in bounded] == [early, early]
    assert [match.chunk_id for match in bounded] == ["p1-c1", "p1-c2"]
    assert bounded[0].score > bounded[1].score
    assert [match.document_id for match in unbounded] == [late, early, early]
    assert unbounded[0].score == pytest.approx(1.0, abs=1e-6)

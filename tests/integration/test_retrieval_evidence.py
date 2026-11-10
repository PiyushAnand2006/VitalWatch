"""Stage 5 gate: real local embeddings + pgvector retrieval compiled into evidence."""

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
from app.schemas.evidence import EvidencePackage, ObservationStatus
from app.schemas.extraction import ExtractedDocument, ExtractedPage, ExtractionMethod
from app.schemas.retrieval import NoEvidenceReason, RetrievalStatus
from app.services.document_storage import LocalDocumentStorage
from app.services.embeddings import LocalEmbeddingService
from app.services.evidence import EvidenceCompiler
from app.services.extraction import chunk_document
from app.services.indexing import DocumentIndexer
from app.services.retrieval import RetrievalService

pytestmark = pytest.mark.integration

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _embeddings() -> LocalEmbeddingService:
    return LocalEmbeddingService(MODEL_NAME, device="cpu", batch_size=4)


async def _index_document(
    sessions: DatabaseSessionManager,
    storage: LocalDocumentStorage,
    embeddings: LocalEmbeddingService,
    *,
    patient_id: str,
    name: str,
    text: str,
    created_at: datetime,
) -> UUID:
    document = await DocumentRepository(sessions.sessionmaker).create(
        DocumentModel(
            patient_id=patient_id,
            uploader_id="synthetic-user",
            source_name=name,
            storage_key=f"{sha256(name.encode()).hexdigest()}.txt",
            media_type="text/plain",
            checksum=sha256(text.encode()).hexdigest(),
            size_bytes=len(text),
            status="processed",
            created_at=created_at,
        )
    )
    extraction = ExtractedDocument(
        pages=[
            ExtractedPage(
                page_number=1, text=text, method=ExtractionMethod.TEXT, quality="high"
            )
        ]
    )
    storage.store(
        f"{document.id}.extraction.json",
        json.dumps(
            {
                "document_id": str(document.id),
                "pages": [page.model_dump(mode="json") for page in extraction.pages],
                "chunks": [chunk.model_dump(mode="json") for chunk in chunk_document(extraction)],
            }
        ).encode("utf-8"),
    )
    vectors = ChunkEmbeddingRepository(sessions.sessionmaker)
    await DocumentIndexer(vectors, storage, embeddings).index_document(patient_id, document.id)
    return document.id


async def _seed(
    sessions: DatabaseSessionManager, storage: LocalDocumentStorage
) -> tuple[LocalEmbeddingService, UUID, UUID, UUID]:
    embeddings = _embeddings()
    first = await _index_document(
        sessions,
        storage,
        embeddings,
        patient_id="synthetic-patient-a",
        name="january-note.txt",
        text=(
            "Progress Note 2026-01-01:\n"
            "Laboratory Results:\n"
            "Sodium 140 mmol/L (ref 135-145 mmol/L)\n"
            "Potassium 4.10 mmol/L\n"
        ),
        created_at=datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
    )
    second = await _index_document(
        sessions,
        storage,
        embeddings,
        patient_id="synthetic-patient-a",
        name="march-note.txt",
        text="Laboratory Results:\nSodium 148 mmol/L\n",
        created_at=datetime(2026, 3, 1, 9, 0, tzinfo=UTC),
    )
    foreign = await _index_document(
        sessions,
        storage,
        embeddings,
        patient_id="synthetic-patient-b",
        name="foreign-note.txt",
        text="Laboratory Results:\nSodium 139 mmol/L\n",
        created_at=datetime(2026, 1, 2, 9, 0, tzinfo=UTC),
    )
    return embeddings, first, second, foreign


def _retrieval(
    embeddings: LocalEmbeddingService, sessions: DatabaseSessionManager
) -> RetrievalService:
    return RetrievalService(
        ChunkEmbeddingRepository(sessions.sessionmaker),
        embeddings,
        top_k=5,
        min_score=0.25,
    )


async def test_retrieval_to_evidence_with_real_embeddings_and_pgvector(
    sessions: DatabaseSessionManager, tmp_path: Path
) -> None:
    embeddings, first_document, second_document, _ = await _seed(
        sessions, LocalDocumentStorage(tmp_path / "documents")
    )
    service = _retrieval(embeddings, sessions)
    as_of = datetime(2026, 2, 1, tzinfo=UTC)

    retrieval = await service.retrieve(
        patient_id="synthetic-patient-a", query="sodium level", as_of=as_of
    )

    assert retrieval.status is RetrievalStatus.RETRIEVED
    assert retrieval.items
    assert {item.document_id for item in retrieval.items} == {first_document}
    assert all(item.patient_id == "synthetic-patient-a" for item in retrieval.items)
    package = EvidenceCompiler().compile(retrieval=retrieval, as_of=as_of)
    assert isinstance(package, EvidencePackage)
    sodium = next(
        observation
        for observation in package.observations
        if observation.canonical_name == "Sodium"
    )
    assert sodium.value_lexeme == "140"
    assert sodium.unit_lexeme == "mmol/L"
    assert sodium.status is ObservationStatus.EXPLICIT
    assert sodium.observed_date is not None
    resolved = package.resolve(sodium.evidence_id)
    assert resolved is not None
    assert resolved.document_id == first_document
    assert package.patient_id == "synthetic-patient-a"

    without_boundary = await service.retrieve(
        patient_id="synthetic-patient-a", query="sodium level"
    )
    assert {item.document_id for item in without_boundary.items} >= {
        first_document,
        second_document,
    }
    conflicting = EvidenceCompiler().compile(retrieval=without_boundary)
    sodium_conflict = next(
        conflict
        for conflict in conflicting.conflicts
        if conflict.canonical_name == "Sodium"
    )
    assert set(sodium_conflict.value_lexemes) == {"140", "148"}


async def test_cross_patient_retrieval_returns_only_authorized_evidence(
    sessions: DatabaseSessionManager, tmp_path: Path
) -> None:
    embeddings, _, _, foreign_document = await _seed(
        sessions, LocalDocumentStorage(tmp_path / "documents")
    )

    retrieval = await _retrieval(embeddings, sessions).retrieve(
        patient_id="synthetic-patient-b", query="sodium level"
    )

    assert retrieval.items
    assert {item.document_id for item in retrieval.items} == {foreign_document}
    assert all("148" not in item.text for item in retrieval.items)
    package = EvidenceCompiler().compile(retrieval=retrieval)
    assert all(
        observation.source.document_id == foreign_document
        for observation in package.observations
    )


async def test_document_filter_limits_candidates(
    sessions: DatabaseSessionManager, tmp_path: Path
) -> None:
    embeddings, first_document, second_document, _ = await _seed(
        sessions, LocalDocumentStorage(tmp_path / "documents")
    )

    retrieval = await _retrieval(embeddings, sessions).retrieve(
        patient_id="synthetic-patient-a",
        query="sodium level",
        document_ids=[second_document],
    )

    assert retrieval.items
    assert {item.document_id for item in retrieval.items} == {second_document}
    assert first_document not in {item.document_id for item in retrieval.items}


async def test_no_evidence_states_with_real_embeddings(
    sessions: DatabaseSessionManager, tmp_path: Path
) -> None:
    embeddings, _, _, _ = await _seed(
        sessions, LocalDocumentStorage(tmp_path / "documents")
    )
    service = _retrieval(embeddings, sessions)

    unknown_patient = await service.retrieve(
        patient_id="synthetic-patient-unknown", query="sodium level"
    )
    strict = RetrievalService(
        ChunkEmbeddingRepository(sessions.sessionmaker), embeddings, min_score=0.999
    )
    strict_result = await strict.retrieve(
        patient_id="synthetic-patient-a", query="sodium level"
    )

    assert unknown_patient.status is RetrievalStatus.NO_EVIDENCE
    assert unknown_patient.reason is NoEvidenceReason.NO_MATCHING_DOCUMENTS
    assert strict_result.status is RetrievalStatus.NO_EVIDENCE
    assert strict_result.reason is NoEvidenceReason.BELOW_RELEVANCE_THRESHOLD
    assert strict_result.items == []


async def test_observations_after_as_of_are_excluded_from_evidence(
    sessions: DatabaseSessionManager, tmp_path: Path
) -> None:
    storage = LocalDocumentStorage(tmp_path / "documents")
    embeddings = _embeddings()
    document_id = await _index_document(
        sessions,
        storage,
        embeddings,
        patient_id="synthetic-patient-a",
        name="future-note.txt",
        text="Laboratory Results 2026-06-01:\nSodium 141 mmol/L\n",
        created_at=datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
    )
    service = _retrieval(embeddings, sessions)
    as_of = datetime(2026, 2, 1, tzinfo=UTC)

    retrieval = await service.retrieve(
        patient_id="synthetic-patient-a", query="sodium level", as_of=as_of
    )
    package = EvidenceCompiler().compile(retrieval=retrieval, as_of=as_of)

    assert retrieval.items
    assert {item.document_id for item in retrieval.items} == {document_id}
    assert package.observations == []
    assert len(package.excluded_after_as_of) == 1
    assert package.excluded_after_as_of[0].canonical_name == "Sodium"

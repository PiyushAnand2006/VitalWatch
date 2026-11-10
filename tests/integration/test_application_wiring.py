"""The application factory exposes Stage 4 services without loading the model eagerly."""

from fastapi.testclient import TestClient

from app.repositories.embeddings import ChunkEmbeddingRepository
from app.services.embeddings import LocalEmbeddingService
from app.services.evidence import EvidenceCompiler
from app.services.indexing import DocumentIndexer
from app.services.retrieval import RetrievalService


def test_stage_four_and_five_services_are_wired_and_model_is_lazy(client: TestClient) -> None:
    state = client.app.state

    assert isinstance(state.vector_repository, ChunkEmbeddingRepository)
    assert isinstance(state.embeddings, LocalEmbeddingService)
    assert isinstance(state.document_indexer, DocumentIndexer)
    assert isinstance(state.retrieval_service, RetrievalService)
    assert isinstance(state.evidence_compiler, EvidenceCompiler)
    assert state.embeddings.model_info is None

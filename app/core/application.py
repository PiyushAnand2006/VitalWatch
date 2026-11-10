"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_v1_router
from app.core.audit import AuditService, LoggingAuditSink
from app.core.config import Settings, get_settings
from app.core.database import get_database_manager
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.request_context import RequestIDMiddleware
from app.privacy.dispatch import ApprovedDispatcher
from app.privacy.engine import PresidioPiiAnalyzer
from app.privacy.gate import PrivacyGate
from app.repositories.documents import DocumentRepository, PatientGrantRepository
from app.repositories.embeddings import ChunkEmbeddingRepository
from app.repositories.privacy import PrivacyApprovalRepository
from app.services.document_processing import DocumentProcessor
from app.services.document_storage import LocalDocumentStorage
from app.services.embeddings import LocalEmbeddingService
from app.services.evidence import EvidenceCompiler
from app.services.extraction import DocumentExtractor
from app.services.indexing import DocumentIndexer
from app.services.retrieval import RetrievalService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize private runtime directories after settings validation."""

    settings: Settings = app.state.settings
    settings.document_storage_directory.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        await app.state.database.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build an independently testable application with explicit dependencies."""

    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings)
    docs_url = "/docs" if resolved_settings.api_docs_enabled else None
    openapi_url = "/openapi.json" if resolved_settings.api_docs_enabled else None

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
        debug=resolved_settings.debug,
        docs_url=docs_url,
        redoc_url=None,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.audit_service = AuditService(LoggingAuditSink())
    database = get_database_manager(resolved_settings)
    app.state.database = database
    app.state.patient_grants = PatientGrantRepository(database.sessionmaker)
    app.state.document_repository = DocumentRepository(database.sessionmaker)
    app.state.document_storage = LocalDocumentStorage(resolved_settings.document_storage_directory)
    app.state.vector_repository = ChunkEmbeddingRepository(database.sessionmaker)
    app.state.embeddings = LocalEmbeddingService(
        model_name=resolved_settings.embedding_model_name,
        revision=resolved_settings.embedding_model_revision,
        cache_directory=resolved_settings.embedding_model_directory,
        device=resolved_settings.embedding_device,
        batch_size=resolved_settings.embedding_batch_size,
        local_files_only=resolved_settings.embedding_local_files_only,
    )
    app.state.document_indexer = DocumentIndexer(
        app.state.vector_repository, app.state.document_storage, app.state.embeddings
    )
    app.state.retrieval_service = RetrievalService(
        app.state.vector_repository,
        app.state.embeddings,
        top_k=resolved_settings.retrieval_top_k,
        min_score=resolved_settings.retrieval_min_score,
        max_query_chars=resolved_settings.retrieval_max_query_chars,
    )
    app.state.evidence_compiler = EvidenceCompiler()
    app.state.privacy_approvals = PrivacyApprovalRepository(database.sessionmaker)
    app.state.privacy_gate = PrivacyGate(
        analyzer=PresidioPiiAnalyzer(
            model_name=resolved_settings.privacy_nlp_model,
            score_threshold=resolved_settings.privacy_analyzer_score_threshold,
        ),
        approvals=app.state.privacy_approvals,
        local_only=resolved_settings.privacy_local_only,
        max_question_chars=resolved_settings.privacy_max_question_chars,
        approval_ttl_seconds=resolved_settings.privacy_approval_ttl_seconds,
    )
    app.state.privacy_dispatcher = ApprovedDispatcher(
        approvals=app.state.privacy_approvals,
        local_only=resolved_settings.privacy_local_only,
    )
    app.state.document_processor = DocumentProcessor(
        app.state.document_repository,
        app.state.document_storage,
        DocumentExtractor(resolved_settings.ocr_language, resolved_settings.tesseract_command),
        app.state.document_indexer,
    )

    app.add_middleware(RequestIDMiddleware)
    register_exception_handlers(app)
    app.include_router(api_v1_router, prefix="/api/v1")
    return app

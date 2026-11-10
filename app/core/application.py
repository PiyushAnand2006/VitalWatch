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
from app.repositories.documents import DocumentRepository, PatientGrantRepository
from app.services.document_storage import LocalDocumentStorage


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

    app.add_middleware(RequestIDMiddleware)
    register_exception_handlers(app)
    app.include_router(api_v1_router, prefix="/api/v1")
    return app

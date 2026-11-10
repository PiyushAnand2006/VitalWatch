"""Authenticated document lifecycle routes; processing also indexes chunk vectors."""

from typing import Annotated, cast
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)

from app.core.auth import AuthenticatedUser, get_current_user
from app.core.exceptions import AppError
from app.models import DocumentModel
from app.schemas.documents import DocumentPage, DocumentStatus, DocumentSummary
from app.services.uploads import UploadValidator, generated_storage_key

router = APIRouter(prefix="/patients/{patient_id}/documents", tags=["documents"])


async def _authorize(request: Request, patient_id: str, user: AuthenticatedUser) -> None:
    if not await request.app.state.patient_grants.allows(user.user_id, patient_id):
        await request.app.state.audit_service.record(
            "document.access",
            "blocked",
            actor_ref=user.user_id,
            resource_type="patient",
            reason_code="not_authorized",
        )
        raise AppError(
            403, "patient_access_denied", "You are not authorized to access this patient."
        )


async def _audited_authorized_user(
    request: Request, patient_id: str, user: Annotated[AuthenticatedUser, Depends(get_current_user)]
) -> AuthenticatedUser:
    await _authorize(request, patient_id, user)
    return user


@router.post(
    "",
    response_model=DocumentSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    patient_id: Annotated[str, Path(min_length=1, max_length=128)],
    file: Annotated[UploadFile, File(description="PDF, TXT, PNG, JPEG, or DOCX document")],
    user: Annotated[AuthenticatedUser, Depends(_audited_authorized_user)],
) -> DocumentSummary:
    data = await file.read()
    validator = UploadValidator(
        request.app.state.settings.upload_max_mb * 1024 * 1024,
        request.app.state.settings.upload_max_pages,
    )
    validated = validator.validate(file.filename, file.content_type, data)
    repository = request.app.state.document_repository
    duplicate = await repository.find_duplicate(patient_id, validated.checksum)
    if duplicate is not None:
        raise AppError(
            409, "duplicate_document", "This document was already uploaded for this patient."
        )
    storage_key = generated_storage_key(validated.suffix)
    storage = request.app.state.document_storage
    try:
        storage.store(storage_key, validated.data)
        document = await repository.create(
            DocumentModel(
                patient_id=patient_id,
                uploader_id=user.user_id,
                source_name=validated.source_name,
                storage_key=storage_key,
                media_type=validated.media_type,
                checksum=validated.checksum,
                size_bytes=len(validated.data),
                status=DocumentStatus.UPLOADED.value,
            )
        )
    except Exception:
        storage.delete(storage_key)
        raise
    await request.app.state.audit_service.record(
        "document.upload", "success", actor_ref=user.user_id, resource_type="document"
    )
    background_tasks.add_task(request.app.state.document_processor.process, patient_id, document.id)
    return cast(DocumentSummary, document)


@router.get("", response_model=DocumentPage, summary="List authorized documents")
async def list_documents(
    request: Request,
    patient_id: Annotated[str, Path(min_length=1, max_length=128)],
    user: Annotated[AuthenticatedUser, Depends(_audited_authorized_user)],
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 25,
) -> DocumentPage:
    items, total = await request.app.state.document_repository.list(patient_id, page, page_size)
    await request.app.state.audit_service.record(
        "document.list", "success", actor_ref=user.user_id, resource_type="document"
    )
    return DocumentPage(items=list(items), page=page, page_size=page_size, total=total)


@router.get(
    "/{document_id}", response_model=DocumentSummary, summary="Get document status and metadata"
)
async def get_document(
    request: Request,
    patient_id: str,
    document_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(_audited_authorized_user)],
) -> DocumentSummary:
    found = await request.app.state.document_repository.get(patient_id, document_id)
    if found is None:
        raise AppError(404, "document_not_found", "The document was not found.")
    await request.app.state.audit_service.record(
        "document.read", "success", actor_ref=user.user_id, resource_type="document"
    )
    return cast(DocumentSummary, found[0])


@router.post(
    "/{document_id}/reprocess",
    response_model=DocumentSummary,
    summary="Queue an explicit document reprocess",
)
async def reprocess_document(
    request: Request,
    background_tasks: BackgroundTasks,
    patient_id: str,
    document_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(_audited_authorized_user)],
) -> DocumentSummary:
    document = await request.app.state.document_repository.update_status(
        patient_id, document_id, DocumentStatus.UPLOADED
    )
    if document is None:
        raise AppError(404, "document_not_found", "The document was not found.")
    await request.app.state.audit_service.record(
        "document.reprocess", "success", actor_ref=user.user_id, resource_type="document"
    )
    background_tasks.add_task(request.app.state.document_processor.process, patient_id, document_id)
    return cast(DocumentSummary, document)


@router.delete(
    "/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a document"
)
async def delete_document(
    request: Request,
    patient_id: str,
    document_id: UUID,
    user: Annotated[AuthenticatedUser, Depends(_audited_authorized_user)],
) -> Response:
    storage_key = await request.app.state.document_repository.delete(patient_id, document_id)
    if storage_key is None:
        raise AppError(404, "document_not_found", "The document was not found.")
    request.app.state.document_storage.delete(storage_key)
    request.app.state.document_storage.delete(f"{document_id}.extraction.json")
    await request.app.state.audit_service.record(
        "document.delete", "success", actor_ref=user.user_id, resource_type="document"
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)

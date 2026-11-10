"""Authenticated provider configuration status without secrets or prompts."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from app.core.auth import AuthenticatedUser, get_current_user
from app.llm.config import describe_providers
from app.schemas.providers import ProvidersResponse

router = APIRouter(tags=["providers"])


@router.get(
    "/providers",
    response_model=ProvidersResponse,
    summary="External model configuration status",
)
async def provider_status(
    request: Request,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
) -> ProvidersResponse:
    settings = request.app.state.settings
    await request.app.state.audit_service.record(
        "provider.status",
        "success",
        actor_ref=user.user_id,
        resource_type="provider",
    )
    return describe_providers(settings, request.app.state.provider_slots)

"""Liveness and readiness endpoints."""

from fastapi import APIRouter, Request, Response, status

from app.core.config import Settings
from app.core.request_context import get_request_id
from app.schemas.health import HealthResponse, ReadyResponse
from app.services.readiness import ReadinessService

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="Service liveness")
async def health(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env.value,
        request_id=get_request_id(),
    )


@router.get("/ready", response_model=ReadyResponse, summary="Dependency readiness")
async def ready(request: Request, response: Response) -> ReadyResponse:
    settings: Settings = request.app.state.settings
    checks = ReadinessService(settings).evaluate()
    is_ready = all(check.status == "passed" for check in checks)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadyResponse(
        status="ready" if is_ready else "not_ready",
        service=settings.app_name,
        version=settings.app_version,
        checks=checks,
        request_id=get_request_id(),
    )

"""Safe application exceptions and HTTP exception handlers."""

import logging
from http import HTTPStatus
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.request_context import get_request_id
from app.schemas.errors import ErrorBody, ErrorResponse, ValidationIssue

logger = logging.getLogger(__name__)


class AppError(Exception):
    """An expected application error safe to expose through its public contract."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _error_response(
    status_code: int,
    code: str,
    message: str,
    issues: list[ValidationIssue] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        error=ErrorBody(code=code, message=message, issues=issues),
        request_id=get_request_id(),
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return expected application failures without internal details."""

    error = cast(AppError, exc)
    return _error_response(error.status_code, error.code, error.message)


async def http_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Normalize framework HTTP errors into the public error envelope."""

    error = cast(HTTPException, exc)
    message = (
        error.detail if isinstance(error.detail, str) else "The request could not be processed."
    )
    code = "not_found" if error.status_code == HTTPStatus.NOT_FOUND else "http_error"
    return _error_response(error.status_code, code, message)


async def validation_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return field locations and types without submitted values or exception context."""

    error = cast(RequestValidationError, exc)
    issues = [
        ValidationIssue(
            location=[str(part) for part in issue.get("loc", ())],
            message=issue.get("msg", "Invalid value."),
            type=issue.get("type", "validation_error"),
        )
        for issue in error.errors()
    ]
    return _error_response(
        HTTPStatus.UNPROCESSABLE_ENTITY,
        "validation_error",
        "Request validation failed.",
        issues,
    )


async def unexpected_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log only the exception type and return no internals to the client."""

    logger.error("Unhandled request error type=%s", type(exc).__name__)
    return _error_response(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "internal_error",
        "An internal error occurred.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register consistent handlers for expected and unexpected failures."""

    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(HTTPException, http_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)

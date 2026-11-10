"""Request ID propagation and response headers."""

import re
from contextvars import ContextVar
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_current_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str:
    """Return the active request ID or a non-request default."""

    return _current_request_id.get() or "system"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a safe request ID to logs, application state, and every response."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = (
            incoming
            if incoming is not None and _REQUEST_ID_PATTERN.fullmatch(incoming)
            else str(uuid4())
        )
        request.state.request_id = request_id
        token = _current_request_id.set(request_id)
        try:
            try:
                response = await call_next(request)
            except Exception as exc:
                # BaseHTTPMiddleware sits inside the outer server-error handler. Handling
                # the failure here ensures the safe response still carries a request ID.
                from app.core.exceptions import unexpected_exception_handler

                response = await unexpected_exception_handler(request, exc)
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            _current_request_id.reset(token)

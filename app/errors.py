"""Custom exceptions and handlers that produce one consistent error shape:

    {"error": {"code": "...", "message": "...", "request_id": "...", "details": ...}}
"""
import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import InterfaceError, OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for expected, client-facing errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class BadRequestError(AppError):
    status_code, code = 400, "bad_request"


class NotFoundError(AppError):
    status_code, code = 404, "not_found"


class DuplicateDocumentError(AppError):
    status_code, code = 409, "duplicate_document"


class PayloadTooLargeError(AppError):
    status_code, code = 413, "payload_too_large"


class UnsupportedMediaTypeError(AppError):
    status_code, code = 415, "unsupported_media_type"


class RateLimitedError(AppError):
    status_code, code = 429, "rate_limited"


class ServiceUnavailableError(AppError):
    """An upstream dependency (LLM, embeddings, vector DB, Postgres) is down."""

    status_code, code = 503, "service_unavailable"


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "error": {"code": code, "message": message, "request_id": _request_id(request)}
    }
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the application."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return _error_response(request, exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Return location/message/type only; never echo submitted input back.
        details = [
            {"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
            for e in exc.errors()
        ]
        return _error_response(request, 422, "validation_error", "Invalid request data.", details)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed"}
        return _error_response(
            request, exc.status_code, codes.get(exc.status_code, "http_error"),
            str(exc.detail), headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(OperationalError)
    @app.exception_handler(InterfaceError)
    async def handle_database_down(request: Request, exc: Exception) -> JSONResponse:
        # Connection failures / timeouts to Postgres: a dependency outage, not a bug.
        logger.error("Database unavailable: %s [request_id=%s]", type(exc).__name__, _request_id(request))
        return _error_response(
            request, 503, "service_unavailable", "The database is temporarily unavailable."
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error [request_id=%s]", _request_id(request))
        return _error_response(
            request, 500, "internal_error", "An unexpected error occurred.",
            headers={"X-Request-ID": _request_id(request)},
        )

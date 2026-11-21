"""Logging setup: per-request IDs, request timing and secret redaction."""
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import Settings

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
logger = logging.getLogger("app.request")

# Only accept safe client-supplied request IDs (prevents log injection).
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class RequestIdFilter(logging.Filter):
    """Attach the current request ID to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


class RedactingFilter(logging.Filter):
    """Mask any known secret value that ends up in a log message."""

    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        if self._secrets:
            message = record.getMessage()
            for secret in self._secrets:
                message = message.replace(secret, "***")
            record.msg, record.args = message, ()
        return True


def configure_logging(settings: Settings) -> None:
    """Configure root logging once at startup."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
        )
    )
    handler.addFilter(RequestIdFilter())
    handler.addFilter(RedactingFilter(settings.secret_values()))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    # Our middleware already logs every request; silence uvicorn's duplicate.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request ID, log method/path/status/latency, echo the ID back."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("X-Request-ID", "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        token = request_id_ctx.set(request_id)
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            # Path only: query strings could carry sensitive data.
            logger.info(
                "%s %s -> %s in %.1f ms",
                request.method, request.url.path, status_code, latency_ms,
            )
            request_id_ctx.reset(token)

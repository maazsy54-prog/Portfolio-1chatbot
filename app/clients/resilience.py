"""Shared timeout + retry wrapper for every external call.

Transient failures (timeouts, connection errors, 429/5xx) are retried with
exponential backoff. Anything that still fails is raised as a single
ServiceUnavailableError (HTTP 503) so routes never leak vendor exceptions.
"""
import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt

from app.config import Settings
from app.errors import ServiceUnavailableError

try:  # pinecone's sync client uses urllib3
    from urllib3.exceptions import HTTPError as _Urllib3Error
except ImportError:  # pragma: no cover
    _Urllib3Error = ()  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)
T = TypeVar("T")

TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def get_status(exc: BaseException | None) -> int | None:
    """Read the HTTP status from a vendor exception (Cohere/Pinecone use `status_code`,
    older SDKs use `status`)."""
    if exc is None:
        return None
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status if isinstance(status, int) else None


def describe_exception(exc: BaseException) -> str:
    """The vendor's own error message if it has one (e.g. a monthly-quota notice), else str(exc)."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body.get("message"):
        return str(body["message"])[:300]
    return str(exc)[:300]


def is_transient(exc: BaseException) -> bool:
    """Return True if retrying the call could plausibly succeed."""
    if isinstance(
        exc, (asyncio.TimeoutError, TimeoutError, ConnectionError,
              httpx.TimeoutException, httpx.TransportError)
    ):
        return True
    if _Urllib3Error and isinstance(exc, _Urllib3Error):
        return True
    return get_status(exc) in TRANSIENT_STATUS_CODES


def make_wait(settings: Settings):
    """Backoff strategy: short exponential waits with jitter, but a rate limit (429) waits
    10s x attempt (max 60s), because a per-minute quota will not clear in a few seconds."""
    backoff = settings.retry_backoff_seconds

    def wait(retry_state) -> float:
        if backoff <= 0:
            return 0.0
        attempt = retry_state.attempt_number
        delay = min(8.0, backoff * 2 ** (attempt - 1)) + random.uniform(0, backoff)
        error = retry_state.outcome.exception() if retry_state.outcome else None
        if get_status(error) == 429:
            delay = max(delay, min(60.0, 10.0 * attempt))
        return delay

    return wait


async def call_with_retries(
    fn: Callable[[], Awaitable[T]], *, service: str, settings: Settings
) -> T:
    """Run `fn` with a per-attempt timeout and retries on transient errors.

    Raises:
        ServiceUnavailableError: the call failed or kept failing.
    """
    wait = make_wait(settings)
    result: T
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max(1, settings.max_retries)),
            wait=wait,
            retry=retry_if_exception(is_transient),
            reraise=True,
        ):
            with attempt:
                result = await asyncio.wait_for(fn(), timeout=settings.http_timeout_seconds)
    except Exception as exc:
        # The client only ever sees a generic 503; the real cause goes to the log.
        # (The logging filter masks any known secret value.)
        logger.error(
            "%s call failed: %s (status=%s): %s", service, type(exc).__name__,
            get_status(exc), describe_exception(exc),
        )
        raise ServiceUnavailableError(f"{service} is temporarily unavailable.") from exc
    return result

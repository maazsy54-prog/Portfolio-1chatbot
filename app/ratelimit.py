"""In-memory sliding-window rate limiting (per client, per endpoint group).

Fine for a single process. With several workers each keeps its own counters,
so production would move this to Redis; noted in DESIGN.md.
"""
import math
import time
from collections import deque
from collections.abc import Callable

from fastapi import Request

from app.errors import RateLimitedError


class SlidingWindowLimiter:
    """Allow at most `limit` calls per `window_seconds` for each (bucket, key)."""

    def __init__(self, window_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = {}

    def check(self, bucket: str, key: str, limit: int) -> None:
        """Record a call, or raise RateLimitedError (429) if over the limit."""
        now = self._clock()
        hits = self._hits.setdefault((bucket, key), deque())
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= limit:
            retry_after = max(1, math.ceil(self._window - (now - hits[0])))
            raise RateLimitedError(
                f"Rate limit exceeded: at most {limit} requests per minute. Try again in {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)
        if len(self._hits) > 10_000:  # keep memory bounded
            self._prune(now)

    def _prune(self, now: float) -> None:
        for k in [k for k, h in self._hits.items() if not h or now - h[-1] >= self._window]:
            del self._hits[k]


def client_key(request: Request) -> str:
    """Identify the caller. X-Forwarded-For is honoured only when explicitly trusted,
    otherwise anyone could dodge the limit by sending a fake header."""
    if request.app.state.settings.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()[:64]
    return request.client.host if request.client else "unknown"


def limit_ask(request: Request) -> None:
    """Dependency: rate limit for /ask."""
    request.app.state.limiter.check(
        "ask", client_key(request), request.app.state.settings.rate_limit_ask_per_minute
    )


def limit_upload(request: Request) -> None:
    """Dependency: rate limit for uploads."""
    request.app.state.limiter.check(
        "upload", client_key(request), request.app.state.settings.rate_limit_upload_per_minute
    )

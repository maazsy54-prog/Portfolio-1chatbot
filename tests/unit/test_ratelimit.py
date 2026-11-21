"""Unit tests for the sliding-window limiter (fake clock, no sleeping)."""
import pytest

from app.errors import RateLimitedError
from app.ratelimit import SlidingWindowLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_allows_up_to_limit_then_blocks_with_retry_after() -> None:
    clock = Clock()
    limiter = SlidingWindowLimiter(60, clock)
    for _ in range(3):
        limiter.check("ask", "1.2.3.4", 3)
    with pytest.raises(RateLimitedError) as info:
        limiter.check("ask", "1.2.3.4", 3)
    assert info.value.status_code == 429
    assert info.value.headers == {"Retry-After": "60"}


def test_window_slides_and_frees_capacity() -> None:
    clock = Clock()
    limiter = SlidingWindowLimiter(60, clock)
    limiter.check("ask", "k", 1)
    clock.now += 30
    with pytest.raises(RateLimitedError) as info:
        limiter.check("ask", "k", 1)
    assert info.value.headers == {"Retry-After": "30"}
    clock.now += 31
    limiter.check("ask", "k", 1)  # allowed again


def test_clients_and_buckets_are_independent() -> None:
    limiter = SlidingWindowLimiter(60, Clock())
    limiter.check("ask", "a", 1)
    limiter.check("ask", "b", 1)       # different client
    limiter.check("upload", "a", 1)    # different bucket
    with pytest.raises(RateLimitedError):
        limiter.check("ask", "a", 1)

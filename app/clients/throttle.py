"""Client-side pacing so we stay under a vendor's calls-per-minute limit.

Trial API keys allow only a few calls a minute. Waiting briefly here is far better than
sending the call, getting a 429 and failing. A limit of 0 means unlimited.
"""
import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable


class CallThrottle:
    """Sliding-window limiter: at most `calls_per_minute` calls in any 60-second window."""

    def __init__(
        self,
        calls_per_minute: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._limit = calls_per_minute
        self._window = window_seconds
        self._clock = clock
        self._sleep = sleep
        self._calls: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait (if needed) until one more call is allowed, then reserve it."""
        if self._limit <= 0:
            return
        async with self._lock:  # waiters are served in order
            while True:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self._window:
                    self._calls.popleft()
                if len(self._calls) < self._limit:
                    self._calls.append(now)
                    return
                await self._sleep(self._window - (now - self._calls[0]) + 0.05)

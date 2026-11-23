"""Unit tests for client-side pacing and the 429-aware backoff (virtual time, no real sleeping)."""
import asyncio
from types import SimpleNamespace

from app.clients.cohere_client import EmbeddingClient, LLMClient, Reranker
from app.clients.resilience import make_wait
from app.clients.throttle import CallThrottle
from app.config import Settings


class VirtualTime:
    """A fake clock whose sleep() just advances time."""

    def __init__(self) -> None:
        self.now, self.sleeps = 0.0, []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def throttle(limit: int, vt: VirtualTime) -> CallThrottle:
    return CallThrottle(limit, clock=vt.clock, sleep=vt.sleep)


async def test_calls_under_the_limit_do_not_wait() -> None:
    vt = VirtualTime()
    t = throttle(3, vt)
    for _ in range(3):
        await t.acquire()
    assert vt.sleeps == [] and vt.now == 0


async def test_call_over_the_limit_waits_for_the_window() -> None:
    vt = VirtualTime()
    t = throttle(2, vt)
    for _ in range(3):
        await t.acquire()
    assert len(vt.sleeps) == 1 and 59.9 < vt.sleeps[0] < 60.2


async def test_many_concurrent_callers_never_exceed_the_limit() -> None:
    vt = VirtualTime()
    t = throttle(2, vt)
    stamps: list[float] = []

    async def call() -> None:
        await t.acquire()
        stamps.append(vt.now)

    await asyncio.gather(*(call() for _ in range(6)))
    for stamp in stamps:  # any 60s window holds at most 2 calls
        assert sum(stamp <= s < stamp + 60 for s in stamps) <= 2
    assert vt.now >= 120  # six calls at two per minute need two full waits


async def test_zero_means_unlimited() -> None:
    vt = VirtualTime()
    t = throttle(0, vt)
    for _ in range(100):
        await t.acquire()
    assert vt.sleeps == []


def retry_state(status: int | None, attempt: int):
    error = SimpleNamespace(status_code=status) if status else ValueError("x")
    return SimpleNamespace(attempt_number=attempt, outcome=SimpleNamespace(exception=lambda: error))


def test_rate_limit_waits_much_longer_than_other_errors() -> None:
    wait = make_wait(Settings(_env_file=None, retry_backoff_seconds=0.5))
    assert wait(retry_state(429, 1)) >= 10
    assert wait(retry_state(429, 3)) >= 30
    assert wait(retry_state(429, 9)) == 60          # capped
    assert wait(retry_state(503, 2)) <= 8           # ordinary transient error: short backoff


def test_no_backoff_means_no_waiting_even_for_429() -> None:
    wait = make_wait(Settings(_env_file=None, retry_backoff_seconds=0.0))
    assert wait(retry_state(429, 3)) == 0


class CountingThrottle:
    def __init__(self) -> None:
        self.acquired = 0

    async def acquire(self) -> None:
        self.acquired += 1


class FakeCohereEverything:
    async def chat(self, **kw):
        part = SimpleNamespace(text='{"answerable": false, "answer": "x", "citations": []}')
        return SimpleNamespace(message=SimpleNamespace(content=[part]),
                               usage=SimpleNamespace(tokens=SimpleNamespace(input_tokens=1, output_tokens=1)))

    async def rerank(self, **kw):
        return SimpleNamespace(results=[SimpleNamespace(index=0, relevance_score=0.5)])

    async def embed(self, **kw):
        return SimpleNamespace(embeddings=SimpleNamespace(float_=[[0.1] * 4 for _ in kw["texts"]]))


SETTINGS = Settings(_env_file=None, embedding_dim=4, retry_backoff_seconds=0.0)


async def test_every_vendor_call_goes_through_its_throttle() -> None:
    fake = FakeCohereEverything()
    llm_t, rr_t, em_t = CountingThrottle(), CountingThrottle(), CountingThrottle()
    await LLMClient(SETTINGS, client=fake, throttle=llm_t).chat_json(system="s", user="u")
    await Reranker(SETTINGS, client=fake, throttle=rr_t).rerank("q", ["a"], 1)
    await EmbeddingClient(SETTINGS, client=fake, throttle=em_t).embed_query("q")
    assert (llm_t.acquired, rr_t.acquired, em_t.acquired) == (1, 1, 1)


async def test_unthrottled_by_default() -> None:
    """With the default limit of 0, clients behave exactly as before."""
    await LLMClient(SETTINGS, client=FakeCohereEverything()).chat_json(system="s", user="u")

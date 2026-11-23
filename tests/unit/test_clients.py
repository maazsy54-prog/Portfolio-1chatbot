"""Unit tests for the Cohere and Pinecone wrappers, using in-memory fakes."""
import asyncio
from types import SimpleNamespace

import pytest

from app.clients.cohere_client import EmbeddingClient
from app.clients.pinecone_client import VectorRecord, VectorStore, clean_metadata
from app.clients.resilience import call_with_retries, is_transient
from app.config import Settings
from app.errors import ServiceUnavailableError

DIM = 4


def make_settings(**overrides) -> Settings:
    base = dict(
        embedding_dim=DIM, retry_backoff_seconds=0.0, max_retries=3,
        http_timeout_seconds=1.0, embed_batch_size=3, upsert_batch_size=2,
        index_visibility_timeout_seconds=0.3, index_poll_interval_seconds=0.01,
        _env_file=None,
    )
    base.update(overrides)
    return Settings(**base)


class ApiError(Exception):
    """Mimics vendor SDK exceptions that carry an HTTP status."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


# ---------- resilience ----------

@pytest.mark.parametrize("exc,expected", [
    (ApiError(429), True), (ApiError(503), True), (ApiError(401), False),
    (ApiError(400), False), (asyncio.TimeoutError(), True), (ConnectionError(), True),
    (ValueError("x"), False),
])
def test_is_transient(exc: Exception, expected: bool) -> None:
    assert is_transient(exc) is expected


async def test_retries_transient_then_succeeds() -> None:
    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ApiError(429)
        return "ok"

    assert await call_with_retries(flaky, service="X", settings=make_settings()) == "ok"
    assert attempts["n"] == 3


async def test_non_transient_fails_fast_as_503() -> None:
    attempts = {"n": 0}

    async def bad_key() -> str:
        attempts["n"] += 1
        raise ApiError(401)

    with pytest.raises(ServiceUnavailableError):
        await call_with_retries(bad_key, service="X", settings=make_settings())
    assert attempts["n"] == 1


async def test_exhausted_retries_raise_503() -> None:
    attempts = {"n": 0}

    async def down() -> str:
        attempts["n"] += 1
        raise ApiError(503)

    with pytest.raises(ServiceUnavailableError):
        await call_with_retries(down, service="X", settings=make_settings())
    assert attempts["n"] == 3


async def test_timeout_is_enforced_and_retried() -> None:
    async def hang() -> None:
        await asyncio.sleep(5)

    settings = make_settings(http_timeout_seconds=0.05, max_retries=2)
    with pytest.raises(ServiceUnavailableError):
        await call_with_retries(hang, service="X", settings=settings)


# ---------- embedding client ----------

class FakeCohere:
    def __init__(self, fail_first: int = 0) -> None:
        self.calls: list[dict] = []
        self.fail_first = fail_first

    async def embed(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.fail_first:
            raise ApiError(429)
        # vector encodes the text length so order can be checked
        vectors = [[float(len(t))] * DIM for t in kwargs["texts"]]
        return SimpleNamespace(embeddings=SimpleNamespace(float_=vectors))


async def test_embed_documents_batches_and_preserves_order() -> None:
    fake = FakeCohere()
    client = EmbeddingClient(make_settings(), client=fake)
    texts = ["a" * n for n in range(1, 9)]  # 8 texts, batch size 3 -> 3 calls
    vectors = await client.embed_documents(texts)
    assert len(fake.calls) == 3
    assert all(len(c["texts"]) <= 3 for c in fake.calls)
    assert all(c["input_type"] == "search_document" for c in fake.calls)
    assert [v[0] for v in vectors] == [float(n) for n in range(1, 9)]


async def test_embed_query_uses_query_input_type() -> None:
    fake = FakeCohere()
    vector = await EmbeddingClient(make_settings(), client=fake).embed_query("hello")
    assert fake.calls[0]["input_type"] == "search_query"
    assert len(vector) == DIM


async def test_embed_retries_rate_limit() -> None:
    fake = FakeCohere(fail_first=2)
    vector = await EmbeddingClient(make_settings(), client=fake).embed_query("hi")
    assert len(fake.calls) == 3 and len(vector) == DIM


async def test_embed_rejects_empty_text() -> None:
    client = EmbeddingClient(make_settings(), client=FakeCohere())
    with pytest.raises(ValueError):
        await client.embed_documents(["fine", "   "])
    with pytest.raises(ValueError):
        await client.embed_query("")


async def test_embed_wrong_dimension_is_503() -> None:
    class WrongDim(FakeCohere):
        async def embed(self, **kwargs):
            return SimpleNamespace(embeddings=SimpleNamespace(float_=[[1.0, 2.0]]))

    with pytest.raises(ServiceUnavailableError):
        await EmbeddingClient(make_settings(), client=WrongDim()).embed_query("x")


# ---------- vector store ----------

class FakeIndex:
    """Minimal in-memory stand-in for a Pinecone index (one namespace dict each)."""

    def __init__(self, visible_after_queries: int = 0) -> None:
        self.data: dict[str, dict[str, dict]] = {}
        self.upsert_batches: list[int] = []
        self.delete_batches: list[int] = []
        self.queries = 0
        self.visible_after = visible_after_queries

    def upsert(self, vectors, namespace=""):
        self.upsert_batches.append(len(vectors))
        self.data.setdefault(namespace, {}).update({v["id"]: v for v in vectors})

    def query(self, vector, top_k, namespace="", filter=None, include_metadata=False):
        self.queries += 1
        items = list(self.data.get(namespace, {}).values())
        if filter:
            items = [i for i in items if all(i["metadata"].get(k) == v for k, v in filter.items())]
        if self.queries <= self.visible_after:
            items = []
        matches = [SimpleNamespace(id=i["id"], score=0.9, metadata=i["metadata"]) for i in items[:top_k]]
        return SimpleNamespace(matches=matches)

    def delete(self, ids=None, delete_all=False, namespace=""):
        if delete_all:
            self.data.pop(namespace, None)
            return
        self.delete_batches.append(len(ids))
        for i in ids:
            self.data.get(namespace, {}).pop(i, None)


def rec(i: int, **meta) -> VectorRecord:
    return VectorRecord(id=f"d:{i}", values=[0.1] * DIM, metadata={"text": f"t{i}", **meta})


def test_clean_metadata_drops_none_and_stringifies_objects() -> None:
    cleaned = clean_metadata({"page": None, "text": "x", "n": 3, "obj": [1, 2]})
    assert cleaned == {"text": "x", "n": 3, "obj": "[1, 2]"}


async def test_upsert_is_batched_and_none_metadata_removed() -> None:
    index = FakeIndex()
    store = VectorStore(make_settings(), index=index)
    await store.upsert([rec(i, page=None) for i in range(5)], namespace="ns")
    assert index.upsert_batches == [2, 2, 1]
    assert "page" not in index.data["ns"]["d:0"]["metadata"]


async def test_query_maps_matches_and_applies_filter() -> None:
    index = FakeIndex()
    store = VectorStore(make_settings(), index=index)
    await store.upsert([rec(1, document_id="A"), rec(2, document_id="B")], namespace="ns")
    hits = await store.query([0.1] * DIM, top_k=5, namespace="ns", metadata_filter={"document_id": "A"})
    assert [h.id for h in hits] == ["d:1"] and hits[0].score == 0.9


async def test_delete_removes_vectors() -> None:
    index = FakeIndex()
    store = VectorStore(make_settings(), index=index)
    await store.upsert([rec(i) for i in range(3)], namespace="ns")
    await store.delete(["d:0", "d:1"], namespace="ns")
    hits = await store.query([0.1] * DIM, top_k=10, namespace="ns")
    assert [h.id for h in hits] == ["d:2"]


async def test_wait_until_visible_handles_eventual_consistency() -> None:
    index = FakeIndex(visible_after_queries=2)
    store = VectorStore(make_settings(), index=index)
    record = rec(1)
    await store.upsert([record], namespace="ns")
    assert await store.wait_until_visible(record, namespace="ns") is True


async def test_wait_until_visible_times_out() -> None:
    index = FakeIndex(visible_after_queries=10_000)
    store = VectorStore(make_settings(), index=index)
    record = rec(1)
    await store.upsert([record], namespace="ns")
    assert await store.wait_until_visible(record, namespace="ns") is False


async def test_vector_db_outage_is_503() -> None:
    class Down(FakeIndex):
        def query(self, *a, **k):
            raise ApiError(503)

    store = VectorStore(make_settings(), index=Down())
    with pytest.raises(ServiceUnavailableError):
        await store.query([0.1] * DIM, top_k=1)


# ---------- namespace deletion (regression: a missing namespace is a 404, not an outage) ----------

class NamespaceIndex:
    """Index whose delete(delete_all=True) fails like Pinecone does for a missing namespace."""

    def __init__(self, error: Exception | None) -> None:
        self.error, self.deleted = error, []

    def delete(self, ids=None, delete_all=False, namespace=""):
        if self.error:
            raise self.error
        self.deleted.append(namespace)


class StatusError(Exception):
    def __init__(self, **attrs) -> None:
        super().__init__("boom")
        self.__dict__.update(attrs)


async def test_delete_namespace_ignores_missing_namespace_status_code() -> None:
    store = VectorStore(make_settings(), index=NamespaceIndex(StatusError(status_code=404)))
    await store.delete_namespace("never-created")  # must not raise


async def test_delete_namespace_ignores_missing_namespace_legacy_status_attr() -> None:
    store = VectorStore(make_settings(), index=NamespaceIndex(StatusError(status=404)))
    await store.delete_namespace("never-created")


async def test_delete_namespace_still_raises_on_real_errors() -> None:
    store = VectorStore(make_settings(), index=NamespaceIndex(StatusError(status_code=401)))
    with pytest.raises(ServiceUnavailableError):
        await store.delete_namespace("x")
    store = VectorStore(make_settings(), index=NamespaceIndex(StatusError(status_code=500)))
    with pytest.raises(ServiceUnavailableError):
        await store.delete_namespace("x")


async def test_delete_namespace_deletes_existing() -> None:
    index = NamespaceIndex(None)
    await VectorStore(make_settings(), index=index).delete_namespace("scratch")
    assert index.deleted == ["scratch"]


def test_real_pinecone_exception_exposes_status_code() -> None:
    """The installed SDK's own 404 must be recognised by our status helper."""
    from pinecone.exceptions import NotFoundException
    from app.clients.resilience import get_status
    assert get_status(NotFoundException()) == 404
    assert get_status(None) is None and get_status(ValueError("x")) is None


async def test_failure_logs_the_real_cause(caplog) -> None:
    async def down() -> None:
        raise ApiError(401)

    with caplog.at_level("ERROR"):
        with pytest.raises(ServiceUnavailableError):
            await call_with_retries(down, service="Vector database", settings=make_settings())
    assert "status=401" in caplog.text and "ApiError" in caplog.text


def test_error_log_uses_the_vendors_own_message() -> None:
    from app.clients.resilience import describe_exception
    quota = StatusError(status_code=429, body={"message": "You are using a Trial key, limited to 1000 API calls / month."})
    assert "1000 API calls / month" in describe_exception(quota)
    assert describe_exception(ValueError("plain")) == "plain"
    assert len(describe_exception(ValueError("x" * 1000))) == 300

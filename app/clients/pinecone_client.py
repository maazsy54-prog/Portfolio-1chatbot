"""Pinecone vector store wrapper.

The official sync client is run in worker threads so the event loop is never
blocked, and every call goes through the shared timeout/retry wrapper.
"""
import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pinecone import Pinecone

from app.clients.resilience import call_with_retries, get_status
from app.errors import ServiceUnavailableError
from app.config import Settings

T = TypeVar("T")
_DELETE_BATCH = 1000


@dataclass(frozen=True)
class VectorRecord:
    """A vector to store: id, embedding and metadata."""

    id: str
    values: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Match:
    """One search hit."""

    id: str
    score: float
    metadata: dict[str, Any]


def clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Make metadata Pinecone-safe: it rejects null values and odd types."""
    cleaned: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        cleaned[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return cleaned


class VectorStore:
    """Upsert, query and delete vectors in one Pinecone index."""

    def __init__(self, settings: Settings, index: Any | None = None) -> None:
        self._settings = settings
        self._index = index  # injectable for tests

    def _get_index(self) -> Any:
        """Create the index handle lazily (this resolves the host over the network)."""
        if self._index is None:
            pc = Pinecone(api_key=self._settings.pinecone_api_key.get_secret_value())
            self._index = pc.Index(self._settings.pinecone_index)
        return self._index

    async def _call(self, fn: Callable[[Any], T]) -> T:
        """Run a blocking index operation in a thread with timeout + retries."""

        async def call() -> T:
            return await asyncio.to_thread(lambda: fn(self._get_index()))

        return await call_with_retries(call, service="Vector database", settings=self._settings)

    def _ns(self, namespace: str | None) -> str:
        return namespace or self._settings.pinecone_namespace

    async def upsert(self, records: list[VectorRecord], namespace: str | None = None) -> None:
        """Store vectors in batches."""
        ns, size = self._ns(namespace), self._settings.upsert_batch_size
        for i in range(0, len(records), size):
            payload = [
                {"id": r.id, "values": r.values, "metadata": clean_metadata(r.metadata)}
                for r in records[i : i + size]
            ]
            await self._call(lambda idx, p=payload: idx.upsert(vectors=p, namespace=ns))

    async def query(
        self,
        vector: list[float],
        *,
        top_k: int,
        namespace: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Match]:
        """Return the top_k most similar vectors (cosine score, higher = closer)."""
        ns = self._ns(namespace)
        response = await self._call(
            lambda idx: idx.query(
                vector=vector, top_k=top_k, namespace=ns,
                filter=metadata_filter, include_metadata=True,
            )
        )
        return [
            Match(id=m.id, score=float(m.score), metadata=dict(m.metadata or {}))
            for m in response.matches
        ]

    async def delete(self, ids: list[str], namespace: str | None = None) -> None:
        """Delete vectors by id (ids that do not exist are ignored)."""
        ns = self._ns(namespace)
        for i in range(0, len(ids), _DELETE_BATCH):
            batch = ids[i : i + _DELETE_BATCH]
            await self._call(lambda idx, b=batch: idx.delete(ids=b, namespace=ns))

    async def delete_namespace(self, namespace: str) -> None:
        """Delete everything in a namespace (used to clean up scratch data)."""
        try:
            await self._call(lambda idx: idx.delete(delete_all=True, namespace=namespace))
        except ServiceUnavailableError as exc:
            if get_status(exc.__cause__) == 404:
                return  # the namespace does not exist yet: nothing to delete
            raise

    async def wait_until_visible(
        self, record: VectorRecord, namespace: str | None = None
    ) -> bool:
        """Poll until a just-upserted vector is searchable (eventual consistency).

        Returns False if it does not appear within the configured timeout.
        """
        deadline = time.monotonic() + self._settings.index_visibility_timeout_seconds
        while True:
            matches = await self.query(record.values, top_k=1, namespace=namespace)
            if any(m.id == record.id for m in matches):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self._settings.index_poll_interval_seconds)

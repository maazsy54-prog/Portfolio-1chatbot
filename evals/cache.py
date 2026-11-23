"""In-run response caches: identical requests are answered once, then reused.

The evaluation compares configurations that often send the very same request (the same question
embedding, the same rerank input, the same prompt). With temperature 0, reusing the answer saves
API calls (important on rate-limited trial keys) without changing what is measured.
Caches live only for one run and are never written to disk.
"""
import json
from typing import Any


class _Cached:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._cache: dict[Any, Any] = {}
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

    def __getattr__(self, name: str) -> Any:  # everything else passes straight through
        return getattr(self._inner, name)

    async def _get(self, key: Any, make) -> Any:
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        value = await make()
        self._cache[key] = value
        return value


class CachedEmbedder(_Cached):
    async def embed_query(self, text: str) -> list[float]:
        return await self._get(text, lambda: self._inner.embed_query(text))


class CachedReranker(_Cached):
    async def rerank(self, query: str, documents: list[str], top_n: int):
        return await self._get((query, tuple(documents), top_n),
                               lambda: self._inner.rerank(query, documents, top_n))


class CachedStore(_Cached):
    async def query(self, vector, *, top_k, namespace=None, metadata_filter=None):
        key = (namespace, tuple(round(x, 5) for x in vector[:32]), top_k,
               json.dumps(metadata_filter, sort_keys=True, default=str))
        return await self._get(key, lambda: self._inner.query(
            vector, top_k=top_k, namespace=namespace, metadata_filter=metadata_filter))


class CachedLLM(_Cached):
    async def chat_json(self, *, system, user, schema=None, model=None):
        key = (system, user, json.dumps(schema, sort_keys=True) if schema else None, model)
        return await self._get(key, lambda: self._inner.chat_json(
            system=system, user=user, schema=schema, model=model))

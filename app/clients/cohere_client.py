"""Async Cohere embedding client (batching, concurrency, retries)."""
import asyncio
from typing import Any

import cohere

from app.clients.resilience import call_with_retries
from app.config import Settings
from app.errors import ServiceUnavailableError


class EmbeddingClient:
    """Embeds documents and queries with Cohere Embed.

    Documents and queries use different `input_type` values; mixing them up
    measurably hurts retrieval quality, so they are separate methods.
    """

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client or cohere.AsyncClientV2(
            api_key=settings.cohere_api_key.get_secret_value(),
            timeout=settings.http_timeout_seconds,
            max_retries=0,  # retries are handled by call_with_retries
        )
        self._semaphore = asyncio.Semaphore(settings.embed_concurrency)

    async def _embed_batch(self, batch: list[str], input_type: str) -> list[list[float]]:
        """Embed one batch (at most `embed_batch_size` texts)."""

        async def call() -> list[list[float]]:
            response = await self._client.embed(
                texts=batch,
                model=self._settings.embedding_model,
                input_type=input_type,
                embedding_types=["float"],
                output_dimension=self._settings.embedding_dim,
            )
            return response.embeddings.float_

        async with self._semaphore:
            vectors = await call_with_retries(
                call, service="Embedding service", settings=self._settings
            )
        expected = self._settings.embedding_dim
        if len(vectors) != len(batch) or any(len(v) != expected for v in vectors):
            raise ServiceUnavailableError("Embedding service returned unexpected output.")
        return vectors

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many passages in parallel batches, preserving input order."""
        if any(not t.strip() for t in texts):
            raise ValueError("Cannot embed empty text.")
        size = self._settings.embed_batch_size
        batches = [texts[i : i + size] for i in range(0, len(texts), size)]
        results = await asyncio.gather(
            *(self._embed_batch(b, "search_document") for b in batches)
        )
        return [vector for batch in results for vector in batch]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        if not text.strip():
            raise ValueError("Cannot embed an empty query.")
        return (await self._embed_batch([text], "search_query"))[0]

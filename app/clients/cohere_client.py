"""Async Cohere clients: embeddings, reranking and the answer LLM.

All three share the retry/timeout wrapper, so vendor failures surface as a
single ServiceUnavailableError (HTTP 503).
"""
import asyncio
from dataclasses import dataclass
from typing import Any

import cohere

from app.clients.resilience import call_with_retries
from app.clients.throttle import CallThrottle
from app.config import Settings
from app.errors import ServiceUnavailableError

# JSON schema the LLM must follow when answering.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answerable", "answer", "citations"],
}


def make_cohere_client(settings: Settings) -> cohere.AsyncClientV2:
    """Build an async Cohere client; its own retries are off (we retry ourselves)."""
    return cohere.AsyncClientV2(
        api_key=settings.cohere_api_key.get_secret_value(),
        timeout=settings.http_timeout_seconds,
        max_retries=0,
    )


class EmbeddingClient:
    """Embeds documents and queries with Cohere Embed.

    Documents and queries use different `input_type` values; mixing them up
    measurably hurts retrieval quality, so they are separate methods.
    """

    def __init__(self, settings: Settings, client: Any | None = None, throttle: Any | None = None) -> None:
        self._settings = settings
        self._client = client or make_cohere_client(settings)
        self._semaphore = asyncio.Semaphore(settings.embed_concurrency)
        self._throttle = throttle or CallThrottle(settings.embed_calls_per_minute)

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
            await self._throttle.acquire()
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


class Reranker:
    """Re-scores candidate passages against the question with Cohere Rerank."""

    def __init__(self, settings: Settings, client: Any | None = None, throttle: Any | None = None) -> None:
        self._settings = settings
        self._client = client or make_cohere_client(settings)
        self._throttle = throttle or CallThrottle(settings.rerank_calls_per_minute)

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[tuple[int, float]]:
        """Return (original_index, relevance_score) pairs, best first."""
        if not documents:
            return []

        async def call() -> list[tuple[int, float]]:
            response = await self._client.rerank(
                model=self._settings.rerank_model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
            )
            return [(r.index, float(r.relevance_score)) for r in response.results]

        await self._throttle.acquire()
        return await call_with_retries(call, service="Rerank service", settings=self._settings)


@dataclass(frozen=True)
class LLMResponse:
    """Raw model output plus token usage (for cost tracking)."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient:
    """Calls the answer model and returns JSON text constrained by ANSWER_SCHEMA."""

    def __init__(self, settings: Settings, client: Any | None = None, throttle: Any | None = None) -> None:
        self._settings = settings
        self._client = client or make_cohere_client(settings)
        self._throttle = throttle or CallThrottle(settings.llm_calls_per_minute)

    async def chat_json(
        self, *, system: str, user: str,
        schema: dict[str, Any] | None = None, model: str | None = None,
    ) -> LLMResponse:
        """Send one system + user message pair and return the JSON reply.

        `schema` and `model` default to the answer schema and answer model; the
        evaluation judge passes its own.
        """

        async def call() -> LLMResponse:
            response = await self._client.chat(
                model=model or self._settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
                response_format=cohere.JsonObjectResponseFormatV2(json_schema=schema or ANSWER_SCHEMA),
            )
            parts = response.message.content or []
            text = "".join(getattr(p, "text", "") or "" for p in parts)
            tokens = getattr(response.usage, "tokens", None)
            return LLMResponse(
                text=text,
                input_tokens=int(getattr(tokens, "input_tokens", 0) or 0),
                output_tokens=int(getattr(tokens, "output_tokens", 0) or 0),
            )

        await self._throttle.acquire()
        return await call_with_retries(call, service="Language model", settings=self._settings)

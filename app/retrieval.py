"""Retrieval: embed the question, search the vector DB, optionally rerank, threshold."""
from dataclasses import dataclass, replace
from typing import Any

from app.clients.cohere_client import EmbeddingClient, Reranker
from app.clients.pinecone_client import Match, VectorStore
from app.config import Settings


@dataclass(frozen=True)
class RetrievedChunk:
    """A passage returned for a question, with everything needed to cite it."""

    id: str
    document_id: str
    filename: str
    position: int
    text: str
    score: float  # vector cosine similarity (always set; shown in the UI)
    page: int | None = None
    section: str | None = None
    rerank_score: float | None = None  # set only when reranking ran


@dataclass(frozen=True)
class RetrievalResult:
    """Outcome of one retrieval."""

    chunks: list[RetrievedChunk]  # final passages for the LLM, best first
    fetched: int  # candidates returned by the vector DB before filtering
    reranked: bool

    @property
    def top_score(self) -> float | None:
        return self.chunks[0].score if self.chunks else None


def chunk_from_match(match: Match) -> RetrievedChunk:
    """Convert a vector-DB match into a RetrievedChunk (Pinecone returns numbers as floats)."""
    md = match.metadata
    return RetrievedChunk(
        id=match.id,
        document_id=str(md.get("document_id", "")),
        filename=str(md.get("filename", "")),
        position=int(md.get("position", 0)),
        text=str(md.get("text", "")),
        score=match.score,
        page=int(md["page"]) if md.get("page") is not None else None,
        section=md.get("section"),
    )


async def retrieve(
    question: str,
    *,
    embedder: EmbeddingClient,
    store: VectorStore,
    reranker: Reranker,
    settings: Settings,
    namespace: str | None = None,
    document_ids: list[str] | None = None,
    use_rerank: bool | None = None,
) -> RetrievalResult:
    """Find the passages that best answer `question`.

    Steps: embed -> vector search (top_k) -> rerank (optional) -> score threshold
    -> keep `context_top_n`. An empty result means "nothing relevant", and the
    caller must refuse instead of calling the LLM.
    """
    vector = await embedder.embed_query(question)
    metadata_filter: dict[str, Any] | None = (
        {"document_id": {"$in": document_ids}} if document_ids else None
    )
    matches = await store.query(
        vector, top_k=settings.retrieval_top_k, namespace=namespace,
        metadata_filter=metadata_filter,
    )
    candidates = [chunk_from_match(m) for m in matches]
    fetched = len(candidates)
    do_rerank = settings.rerank_enabled if use_rerank is None else use_rerank

    if do_rerank and candidates:
        ranked = await reranker.rerank(
            question, [c.text for c in candidates], top_n=settings.context_top_n
        )
        chunks = [replace(candidates[i], rerank_score=score) for i, score in ranked]
        chunks = [c for c in chunks if (c.rerank_score or 0.0) >= settings.rerank_threshold]
    else:
        chunks = [c for c in candidates if c.score >= settings.similarity_threshold]
        chunks = chunks[: settings.context_top_n]

    return RetrievalResult(chunks=chunks, fetched=fetched, reranked=do_rerank and fetched > 0)

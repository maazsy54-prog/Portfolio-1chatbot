"""Question answering service: guards, retrieval, generation, logging."""
import logging
import time
from dataclasses import replace

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.components import Components
from app.config import Settings
from app.db.models import PUBLIC_OWNER, STATUS_READY, Document, QueryLog
from app.errors import NotFoundError
from app.generation import AnswerResult, generate_answer
from app.guardrails import BLOCKED_INPUT_TEXT, check_input
from app.retrieval import RetrievalResult, retrieve
from app.schemas import AskResponse, ChunkOut, CitationOut, UsageOut

logger = logging.getLogger(__name__)

NO_DOCUMENTS_TEXT = "No documents have been uploaded yet. Upload a document first, then ask again."


def _to_response(
    retrieval: RetrievalResult, result: AnswerResult, request_id: str, latency_ms: float,
    guardrail: str | None = None,
) -> AskResponse:
    cited_ids = {c.chunk.id for c in result.citations}
    return AskResponse(
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        guardrail=guardrail,
        citations=[
            CitationOut(
                number=c.number, chunk_id=c.chunk.id, document_id=c.chunk.document_id,
                filename=c.chunk.filename, page=c.chunk.page, section=c.chunk.section,
            )
            for c in result.citations
        ],
        retrieved_chunks=[
            ChunkOut(
                id=c.id, document_id=c.document_id, filename=c.filename, page=c.page,
                section=c.section, position=c.position, text=c.text, score=c.score,
                rerank_score=c.rerank_score, cited=c.id in cited_ids,
            )
            for c in retrieval.chunks
        ],
        chunks_fetched=retrieval.fetched,
        chunks_returned=len(retrieval.chunks),
        reranked=retrieval.reranked,
        usage=UsageOut(
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            llm_calls=result.llm_calls,
        ),
        request_id=request_id,
        latency_ms=round(latency_ms, 1),
    )


async def _log_query(session: AsyncSession, question: str, response: AskResponse) -> None:
    """Record the request for dashboards. Logging must never fail a request."""
    try:
        session.add(QueryLog(
            request_id=response.request_id, question=question[:2000], refused=response.refused,
            refusal_reason=response.refusal_reason, chunks_fetched=response.chunks_fetched,
            latency_ms=response.latency_ms, input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ))
        await session.commit()
    except Exception:
        logger.exception("Could not write query log")
        await session.rollback()


async def answer_question(
    session: AsyncSession,
    *,
    question: str,
    document_ids: list[str] | None,
    use_rerank: bool | None,
    components: Components,
    settings: Settings,
    request_id: str,
    owner_id: str = PUBLIC_OWNER,
    namespace: str | None = None,
) -> AskResponse:
    """Answer a question from the user's ready documents."""
    start = time.perf_counter()

    verdict = check_input(question)
    if verdict.blocked:  # attack-shaped input: no database, embedding, vector or LLM work at all
        logger.warning("Input blocked [%s/%s] request_id=%s", verdict.category, verdict.rule, request_id)
        blocked = AnswerResult(BLOCKED_INPUT_TEXT, refused=True, refusal_reason="input_blocked")
        response = _to_response(
            RetrievalResult(chunks=[], fetched=0, reranked=False), blocked, request_id,
            (time.perf_counter() - start) * 1000, guardrail=verdict.category,
        )
        await _log_query(session, question, response)
        return response

    ready_ids = set((await session.execute(
        select(Document.id).where(Document.owner_id == owner_id, Document.status == STATUS_READY)
    )).scalars())

    if document_ids:
        unknown = set(document_ids) - ready_ids
        if unknown:
            raise NotFoundError("One or more documents were not found.", details=sorted(unknown))

    if not ready_ids:  # nothing to search: no embedding, vector or LLM calls at all
        empty = RetrievalResult(chunks=[], fetched=0, reranked=False)
        result = AnswerResult(NO_DOCUMENTS_TEXT, refused=True, refusal_reason="no_documents")
        response = _to_response(empty, result, request_id, (time.perf_counter() - start) * 1000)
        await _log_query(session, question, response)
        return response

    retrieval = await retrieve(
        question, embedder=components.embedder, store=components.store,
        reranker=components.reranker, settings=settings, namespace=namespace,
        document_ids=document_ids, use_rerank=use_rerank,
    )
    # Safety net: drop any vector whose document is not `ready` in Postgres
    # (for example one that was deleted while the vector delete was still in flight).
    live = [c for c in retrieval.chunks if c.document_id in ready_ids]
    retrieval = replace(retrieval, chunks=live)

    result = await generate_answer(question, live, llm=components.llm, settings=settings)
    response = _to_response(retrieval, result, request_id, (time.perf_counter() - start) * 1000)
    await _log_query(session, question, response)
    return response

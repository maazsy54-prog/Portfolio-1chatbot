"""Evaluation runner: ingest the corpus per chunking config, answer the golden set, score it.

Backends are injected, so tests run it with in-memory fakes and the CLI with the real clients.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.errors import AppError
from app.generation import format_context, generate_answer
from app.ingestion.chunking import chunk_document
from app.ingestion.extract import extract_text
from app.ingestion.records import build_records
from app.ingestion.validate import validate_upload
from app.retrieval import retrieve
from evals import metrics as m
from evals.configs import EvalConfig
from evals.judge import Judge

logger = logging.getLogger("evals")
KS = (1, 3, 5)


@dataclass
class Backends:
    """The external services the evaluation talks to."""

    embedder: Any
    reranker: Any
    llm: Any
    store: Any
    judge: Judge | None = None


async def ingest_corpus(
    corpus_dir: Path, namespace: str, size: int, overlap: int, b: Backends, settings: Settings
) -> dict[str, int]:
    """Chunk, embed and store every PDF/DOCX in `corpus_dir` under one namespace."""
    await b.store.delete_namespace(namespace)  # always start clean
    files, n_chunks = 0, 0
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".docx"}:
            continue
        data = path.read_bytes()
        try:
            upload = validate_upload(path.name, data, settings.max_upload_mb * 1024 * 1024)
            extracted = extract_text(upload.file_type, data, max_pages=settings.max_pdf_pages)
        except AppError as exc:
            logger.warning("skipping %s: %s", path.name, exc.message)
            continue
        chunks = chunk_document(extracted, size=size, overlap=overlap)
        vectors = await b.embedder.embed_documents([c.text for c in chunks])
        records = build_records(upload.sha256[:12], upload.filename, chunks, vectors)
        await b.store.upsert(records, namespace=namespace)
        await b.store.wait_until_visible(records[-1], namespace=namespace)
        files, n_chunks = files + 1, n_chunks + len(chunks)
    return {"files": files, "chunks": n_chunks}


async def _evaluate_question(
    q: dict, config: EvalConfig, b: Backends, settings: Settings, use_judge: bool
) -> dict:
    """Retrieve, answer and score one golden question."""
    answerable = q["answerable"]
    out: dict[str, Any] = {
        "id": q["id"], "type": q["type"], "question": q["question"], "answerable": answerable,
        "error": None,
    }
    try:
        started = time.perf_counter()
        retrieval = await retrieve(
            q["question"], embedder=b.embedder, store=b.store, reranker=b.reranker,
            settings=settings, namespace=config.namespace, use_rerank=config.rerank,
        )
        result = await generate_answer(q["question"], retrieval.chunks, llm=b.llm, settings=settings)
        out["latency_s"] = round(time.perf_counter() - started, 3)
    except AppError as exc:
        out["error"] = f"{exc.code}: {exc.message}"
        return out

    chunks = retrieval.chunks
    texts, files = [c.text for c in chunks], [c.filename for c in chunks]
    document, evidence = q.get("document"), q.get("evidence", [])
    flags = [m.is_relevant(c.text, c.filename, document, evidence) for c in chunks]
    cited_flags = [m.is_relevant(c.chunk.text, c.chunk.filename, document, evidence) for c in result.citations]

    out.update(
        retrieved=[
            {"id": c.id, "filename": c.filename, "page": c.page, "score": round(c.score, 4),
             "rerank_score": None if c.rerank_score is None else round(c.rerank_score, 4),
             "relevant": f}
            for c, f in zip(chunks, flags)
        ],
        chunks_fetched=retrieval.fetched,
        top_vector_score=max((c.score for c in chunks), default=None),
        top_rerank_score=chunks[0].rerank_score if chunks else None,
        answer=result.answer, refused=result.refused, refusal_reason=result.refusal_reason,
        cited=[c.chunk.id for c in result.citations],
        input_tokens=result.input_tokens, output_tokens=result.output_tokens,
        correct=None, faithful=None, judge_reason=None, judge_tokens=0,
    )
    if answerable:
        for k in KS:
            out[f"hit@{k}"] = m.hit_at_k(flags, k)
        out["recall@3"] = m.recall_at_k(texts, files, document, evidence, 3)
        out["recall@5"] = m.recall_at_k(texts, files, document, evidence, 5)
        out["rr"] = m.reciprocal_rank(flags)
        out["source_page_ok"] = bool(chunks) and chunks[0].filename == document and \
            chunks[0].page in q.get("expected_pages", [])
        out["fact_recall"] = 0.0 if result.refused else m.fact_recall(result.answer, q.get("key_facts", []))
        out["citation_precision"] = m.citation_precision(cited_flags)
        if result.refused:
            out["correct"] = False  # a refusal on an answerable question is a miss
        elif use_judge and b.judge:
            try:
                verdict = await b.judge.evaluate(
                    q["question"], q["reference_answer"],
                    format_context(chunks, settings.max_context_chars)[0], result.answer,
                )
                out.update(correct=verdict.correct, faithful=verdict.faithful,
                           judge_reason=verdict.reason,
                           judge_tokens=verdict.input_tokens + verdict.output_tokens)
            except (AppError, ValueError) as exc:
                out["judge_reason"] = f"judge failed: {exc}"
    return out


async def evaluate_config(
    config: EvalConfig, golden: dict, b: Backends, settings: Settings,
    concurrency: int = 3, use_judge: bool = True,
) -> list[dict]:
    """Answer and score every golden question under one configuration."""
    cfg_settings = settings.model_copy(update={
        "chunk_size_tokens": config.chunk_size, "chunk_overlap_tokens": config.chunk_overlap,
        "rerank_enabled": config.rerank, "similarity_threshold": config.similarity_threshold,
        "rerank_threshold": config.rerank_threshold,
    })
    gate = asyncio.Semaphore(concurrency)

    async def run(q: dict) -> dict:
        async with gate:
            return await _evaluate_question(q, config, b, cfg_settings, use_judge)

    return list(await asyncio.gather(*(run(q) for q in golden["questions"])))


def _mean_of(rows: list[dict], key: str) -> float | None:
    return m.mean([r.get(key) for r in rows])


def _bool_rate(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def summarise(results: list[dict], config: EvalConfig) -> dict:
    """Aggregate per-question results into the headline numbers."""
    ok = [r for r in results if not r["error"]]
    ans = [r for r in ok if r["answerable"]]
    unans = [r for r in ok if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    latencies = [r["latency_s"] for r in ok]

    by_type = {}
    for typ in sorted({r["type"] for r in ans}):
        rows = [r for r in ans if r["type"] == typ]
        by_type[typ] = {"n": len(rows), "hit@5": _mean_of(rows, "hit@5"),
                        "recall@5": _mean_of(rows, "recall@5"), "correct": _bool_rate(rows, "correct")}

    summary = {
        "n_questions": len(results), "n_errors": len(results) - len(ok),
        "n_answerable": len(ans), "n_unanswerable": len(unans),
        "retrieval": {
            **{f"hit@{k}": _mean_of(ans, f"hit@{k}") for k in KS},
            "recall@3": _mean_of(ans, "recall@3"), "recall@5": _mean_of(ans, "recall@5"),
            "mrr": _mean_of(ans, "rr"), "source_page_acc": _bool_rate(ans, "source_page_ok"),
        },
        "answers": {
            "correct_rate": _bool_rate(ans, "correct"),
            "faithful_rate": _bool_rate(answered, "faithful"),
            "fact_recall": _mean_of(ans, "fact_recall"),
            "citation_precision": _mean_of(answered, "citation_precision"),
            "false_refusal_rate": (sum(r["refused"] for r in ans) / len(ans)) if ans else None,
            "unanswerable_refused_rate": (sum(r["refused"] for r in unans) / len(unans)) if unans else None,
            "unanswerable_hallucinated_rate": (sum(not r["refused"] for r in unans) / len(unans)) if unans else None,
        },
        "by_type": by_type,
        "latency_s": {"p50": m.percentile(latencies, 50), "p95": m.percentile(latencies, 95)},
        "tokens_per_question": {
            "generation_in": m.mean([r["input_tokens"] for r in ok]),
            "generation_out": m.mean([r["output_tokens"] for r in ok]),
            "judge_total": m.mean([r["judge_tokens"] for r in ok]),
        },
    }
    if config.similarity_threshold == 0 and config.rerank_threshold == 0:
        summary["calibration"] = {
            "vector_score": m.calibrate_threshold(
                [r["top_vector_score"] for r in ans], [r["top_vector_score"] for r in unans]),
        }
        if config.rerank:
            summary["calibration"]["rerank_score"] = m.calibrate_threshold(
                [r["top_rerank_score"] for r in ans], [r["top_rerank_score"] for r in unans])
    return summary


def find_failures(results: list[dict]) -> list[dict]:
    """List what went wrong, per question: the raw material for the failure analysis."""
    failures = []
    for r in results:
        issues = []
        if r["error"]:
            issues.append("error")
        elif r["answerable"]:
            if r.get("hit@5") == 0.0:
                issues.append("retrieval_miss")
            if r["refused"]:
                issues.append("false_refusal")
            elif r.get("correct") is False:
                issues.append("wrong_answer")
            if r.get("faithful") is False:
                issues.append("unfaithful")
        elif not r["refused"]:
            issues.append("hallucinated_on_unanswerable")
        if issues:
            failures.append({"id": r["id"], "type": r["type"], "question": r["question"],
                             "issues": issues, "answer": r.get("answer"),
                             "refusal_reason": r.get("refusal_reason")})
    return failures

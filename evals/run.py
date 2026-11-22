"""One command runs the whole evaluation and saves the results.

    python -m evals.run                      # all configurations
    python -m evals.run --configs B C        # a subset
    python -m evals.run --no-judge           # retrieval + lexical metrics only (cheap, no judge calls)

Outputs (evals/results/<timestamp>/): results.json, summary.md, per_question.csv, judge_sample.csv
"""
import argparse
import asyncio
import hashlib
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings, get_settings
from evals.configs import CONFIGS, EvalConfig
from evals.judge import Judge
from evals.report import pick_winner, write_outputs
from evals.runner import Backends, evaluate_config, find_failures, ingest_corpus, summarise
from evals.validate_golden import validate_golden

logger = logging.getLogger("evals")


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


async def run_all(
    backends: Backends, golden: dict, configs: list[EvalConfig], corpus_dir: Path,
    settings: Settings, out_dir: Path, concurrency: int = 3, keep_namespaces: bool = False,
    use_judge: bool = True, golden_bytes: bytes = b"",
) -> dict[str, Path]:
    """Ingest per chunking config, evaluate every config, write all outputs."""
    namespaces = {c.namespace: c for c in configs}  # one ingestion per distinct chunking
    ingested: dict[str, dict] = {}
    try:
        for ns, c in namespaces.items():
            logger.info("ingesting corpus into %s (chunks %s/%s)", ns, c.chunk_size, c.chunk_overlap)
            ingested[ns] = await ingest_corpus(corpus_dir, ns, c.chunk_size, c.chunk_overlap, backends, settings)

        all_results, summaries, failures = {}, {}, {}
        for c in configs:
            logger.info("evaluating config %s (%s)", c.name, c.description)
            results = await evaluate_config(c, golden, backends, settings, concurrency, use_judge)
            all_results[c.name] = results
            summaries[c.name] = summarise(results, c)
            failures[c.name] = find_failures(results)
    finally:
        if not keep_namespaces:
            for ns in namespaces:
                try:
                    await backends.store.delete_namespace(ns)
                except Exception:
                    logger.warning("could not delete namespace %s", ns)

    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": _git_commit(), "python": platform.python_version(),
        "golden_questions": len(golden["questions"]),
        "golden_sha256": hashlib.sha256(golden_bytes or json.dumps(golden, sort_keys=True).encode()).hexdigest(),
        "embedding_model": settings.embedding_model, "embedding_dim": settings.embedding_dim,
        "llm_model": settings.llm_model, "judge_model": settings.judge_model or f"{settings.llm_model} (same as LLM)",
        "rerank_model": settings.rerank_model, "temperature": settings.llm_temperature,
        "retrieval_top_k": settings.retrieval_top_k, "context_top_n": settings.context_top_n,
        "ingested": ingested, "judge_used": use_judge,
    }
    paths = write_outputs(out_dir, meta, configs, all_results, summaries, failures)
    paths["winner"] = Path(pick_winner(summaries))  # name only, for the caller to print
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation.")
    parser.add_argument("--configs", nargs="*", help="config names to run (default: all)")
    parser.add_argument("--corpus", default="corpus", type=Path)
    parser.add_argument("--golden", default="evals/golden_set.json", type=Path)
    parser.add_argument("--output", default="evals/results", type=Path)
    parser.add_argument("--concurrency", type=int, default=3, help="parallel questions (lower if rate-limited)")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--keep-namespaces", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from app.clients.cohere_client import EmbeddingClient, LLMClient, Reranker
    from app.clients.pinecone_client import VectorStore

    golden_bytes = args.golden.read_bytes()
    golden = json.loads(golden_bytes)
    problems = validate_golden(golden, args.corpus)
    if problems:
        raise SystemExit("Golden set is invalid:\n" + "\n".join(problems))

    configs = [c for c in CONFIGS if not args.configs or c.name in args.configs]
    if not configs:
        raise SystemExit(f"No matching configs. Available: {[c.name for c in CONFIGS]}")

    # Patient retries: evaluation makes many calls and trial API keys are rate-limited.
    base = get_settings()
    settings = base.model_copy(update={"max_retries": 6, "retry_backoff_seconds": 2.0,
                                       "http_timeout_seconds": 60.0})
    llm = LLMClient(settings)
    backends = Backends(
        embedder=EmbeddingClient(settings), reranker=Reranker(settings), llm=llm,
        store=VectorStore(settings), judge=Judge(llm, settings.judge_model or None),
    )
    out_dir = args.output / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    paths = asyncio.run(run_all(
        backends, golden, configs, args.corpus, settings, out_dir, args.concurrency,
        args.keep_namespaces, not args.no_judge, golden_bytes,
    ))
    print("\n" + paths["summary"].read_text(encoding="utf-8").split("## Failures")[0])
    print(f"Winner by rule: {paths['winner']}\nFull results: {out_dir}")


if __name__ == "__main__":
    main()

"""Connectivity check for every external service, one line each.

Run from the repo root:  python -m scripts.check_services
Prints OK/FAIL per service with the provider's own error text (secrets are masked).
Note: it makes one tiny call to each of embeddings, rerank and the LLM.
"""
import asyncio

from app.config import get_settings
from app.clients.resilience import describe_exception, get_status

settings = get_settings()


def _redact(text: str) -> str:
    for secret in settings.secret_values():
        text = text.replace(secret, "***")
    return text


def _report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {_redact(detail)}")
    return ok


def _fail(name: str, exc: BaseException) -> bool:
    cause = exc.__cause__ or exc  # unwrap our 503 wrapper to show the vendor's message
    return _report(name, False, f"{type(cause).__name__} (HTTP {get_status(cause)}): {describe_exception(cause)}")


async def check_embeddings() -> bool:
    try:
        from app.clients.cohere_client import EmbeddingClient

        vector = await EmbeddingClient(settings).embed_query("connectivity check")
        return _report("Cohere embeddings", len(vector) == settings.embedding_dim,
                       f"model={settings.embedding_model} dim={len(vector)}")
    except Exception as exc:
        return _fail("Cohere embeddings", exc)


async def check_rerank() -> bool:
    try:
        from app.clients.cohere_client import Reranker

        ranked = await Reranker(settings).rerank("shipping time", ["Shipping takes five days.", "Unrelated."], 1)
        return _report("Cohere rerank", bool(ranked), f"model={settings.rerank_model}")
    except Exception as exc:
        return _fail("Cohere rerank", exc)


async def check_llm() -> bool:
    label = f"LLM ({settings.llm_provider}, model={settings.llm_model})"
    try:
        from app.clients.llm import make_llm

        reply = await make_llm(settings).chat_json(
            system="Reply with JSON only.",
            user='Return exactly: {"answerable": true, "answer": "ok", "citations": []}',
        )
        return _report(label, bool(reply.text.strip()), f"replied: {reply.text.strip()[:60]!r}")
    except Exception as exc:
        return _fail(label, exc)


def check_pinecone() -> bool:
    try:
        from pinecone import Pinecone

        pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
        info = pc.describe_index(settings.pinecone_index)
        ok = info.dimension == settings.embedding_dim and info.metric == "cosine"
        return _report("Pinecone index", ok,
                       f"name={settings.pinecone_index} dim={info.dimension} "
                       f"metric={info.metric} (expected dim={settings.embedding_dim}, cosine)")
    except Exception as exc:
        return _fail("Pinecone index", exc)


async def check_neon() -> bool:
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(settings.database_url.get_secret_value())
        try:
            async with engine.connect() as conn:
                version = (await conn.execute(text("SELECT version()"))).scalar_one()
        finally:
            await engine.dispose()
        return _report("Neon Postgres", True, str(version)[:60])
    except Exception as exc:
        return _fail("Neon Postgres", exc)


async def main() -> bool:
    results = [await check_embeddings(), await check_rerank(), await check_llm(),
               check_pinecone(), await check_neon()]
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if asyncio.run(main()) else 1)

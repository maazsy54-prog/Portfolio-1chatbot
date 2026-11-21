"""Connectivity check for Cohere, Pinecone and Neon.

Run from the repo root:  python -m scripts.check_services
Prints OK/FAIL per service. Never prints secret values.
"""
import asyncio

from app.config import get_settings

settings = get_settings()


def _redact(text: str) -> str:
    for secret in settings.secret_values():
        text = text.replace(secret, "***")
    return text


def _report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {_redact(detail)}")
    return ok


def check_cohere() -> bool:
    try:
        import cohere

        client = cohere.ClientV2(settings.cohere_api_key.get_secret_value())
        response = client.embed(
            texts=["connectivity check"],
            model=settings.embedding_model,
            input_type="search_document",
            embedding_types=["float"],
            output_dimension=settings.embedding_dim,
        )
        dim = len(response.embeddings.float_[0])
        return _report("Cohere embed", dim == settings.embedding_dim,
                       f"model={settings.embedding_model} dim={dim}")
    except Exception as exc:
        return _report("Cohere embed", False, f"{type(exc).__name__}: {exc}")


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
        return _report("Pinecone index", False, f"{type(exc).__name__}: {exc}")


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
        return _report("Neon Postgres", False, f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    results = [check_cohere(), check_pinecone(), asyncio.run(check_neon())]
    raise SystemExit(0 if all(results) else 1)

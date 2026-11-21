"""Bundle of long-lived service objects, created once at startup."""
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.clients.cohere_client import EmbeddingClient, LLMClient, Reranker
from app.clients.pinecone_client import VectorStore
from app.config import Settings
from app.db.session import build_engine, build_session_factory


@dataclass
class Components:
    """Everything the services need. Tests construct this with fakes."""

    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    embedder: Any
    reranker: Any
    llm: Any
    store: Any


def build_components(settings: Settings) -> Components:
    """Create the real clients from settings."""
    url = settings.database_url.get_secret_value()
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    engine = build_engine(url)
    return Components(
        engine=engine,
        session_factory=build_session_factory(engine),
        embedder=EmbeddingClient(settings),
        reranker=Reranker(settings),
        llm=LLMClient(settings),
        store=VectorStore(settings),
    )

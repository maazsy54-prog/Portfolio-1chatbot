"""Async engine and session factory."""
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base


def build_engine(url: str) -> AsyncEngine:
    """Create the async engine.

    For Postgres/Neon: pre-ping and recycle connections (Neon suspends idle
    compute, which silently kills pooled connections) and allow time for a
    cold start on the first connection.
    """
    if url.startswith("sqlite"):
        return create_async_engine(url)
    return create_async_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_size=5,
        max_overflow=5,
        connect_args={"timeout": 15},
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory; objects stay usable after commit (no implicit reloads)."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """Create tables if missing. Alembic migrations are the production upgrade path."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

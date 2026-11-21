"""Application entry point. Routing and wiring only; no business logic."""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.errors import register_exception_handlers
from app.logging_conf import RequestContextMiddleware, configure_logging
from app.routers import health

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hook (DB and API clients are opened/closed here later)."""
    logger.info("Starting %s", app.title)
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Upload documents and ask grounded, cited questions about them.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    return app


app = create_app()

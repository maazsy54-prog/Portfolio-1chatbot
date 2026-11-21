"""Application entry point. Routing and wiring only; no business logic."""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.components import Components, build_components
from app.config import Settings, get_settings
from app.db.session import init_db
from app.errors import register_exception_handlers
from app.logging_conf import RequestContextMiddleware, configure_logging
from app.routers import ask, documents, health

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None, components: Components | None = None
) -> FastAPI:
    """Build the FastAPI application. Tests pass fake components."""
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Open clients and the database at startup; close them at shutdown."""
        logger.info("Starting %s", app.title)
        app.state.settings = settings
        app.state.components = components or build_components(settings)
        await init_db(app.state.components.engine)
        yield
        await app.state.components.engine.dispose()
        logger.info("Shut down")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Upload documents and ask grounded, cited questions about them.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(ask.router)
    return app


app = create_app()

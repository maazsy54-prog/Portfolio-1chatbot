"""Application entry point. Routing and wiring only; no business logic."""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.components import Components, build_components
from app.config import Settings, get_settings
from app.db.session import init_db
from app.errors import register_exception_handlers
from app.ratelimit import SlidingWindowLimiter
from app.logging_conf import RequestContextMiddleware, configure_logging
from app.routers import ask, documents, health

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


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
    app.state.settings = settings
    app.state.limiter = SlidingWindowLimiter()
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(ask.router)

    has_frontend = (FRONTEND_DIR / "index.html").exists()
    if has_frontend:  # served from /app so API routes and their JSON 404s are untouched
        app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Send visitors to the web UI (or the API docs if the UI is not installed)."""
        return RedirectResponse("/app/" if has_frontend else "/docs")

    return app


app = create_app()

"""System routes."""
from typing import Annotated

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas import HealthResponse

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report that the service is running and which integrations are configured.

    This does not call external services yet; a readiness check that pings
    Neon, Pinecone and Cohere is added in a later phase.
    """
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        configured={
            "cohere": bool(settings.cohere_api_key.get_secret_value()),
            "pinecone": bool(settings.pinecone_api_key.get_secret_value()),
            "database": bool(settings.database_url.get_secret_value()),
        },
    )

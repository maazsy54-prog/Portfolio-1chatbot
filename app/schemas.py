"""Pydantic request and response models."""
from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Liveness information about the running service."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "app": "Production RAG Service",
                "version": "0.1.0",
                "environment": "development",
                "configured": {"cohere": True, "pinecone": True, "database": True},
            }
        }
    )

    status: str
    app: str
    version: str
    environment: str
    configured: dict[str, bool]

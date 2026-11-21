"""Central application settings.

All configuration lives here. Secrets are read from environment variables
(or a local, git-ignored .env file) and are never hard-coded.
"""
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- App ---
    app_name: str = "Production RAG Service"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"

    # --- Secrets (set in .env, never in code) ---
    cohere_api_key: SecretStr = SecretStr("")
    pinecone_api_key: SecretStr = SecretStr("")
    database_url: SecretStr = SecretStr("")

    # --- Vector DB ---
    pinecone_index: str = "rag-portfolio"
    pinecone_namespace: str = "default"

    # --- Models (verify exact IDs in the Cohere docs) ---
    embedding_model: str = "embed-v4.0"
    embedding_dim: int = 1024  # must match the Pinecone index dimension
    rerank_model: str = "rerank-v3.5"
    llm_model: str = "command-a-03-2025"

    # --- Chunking ---
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50

    # --- Retrieval (threshold is calibrated later from eval data) ---
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    similarity_threshold: float = 0.0

    # --- Limits ---
    max_upload_mb: int = 10
    max_pdf_pages: int = 300
    max_question_chars: int = 1000
    rate_limit: str = "30/minute"

    # --- External call resilience ---
    http_timeout_seconds: float = 30.0
    max_retries: int = 3

    def secret_values(self) -> list[str]:
        """Return all non-empty secret values (used by the log redactor)."""
        secrets = (self.cohere_api_key, self.pinecone_api_key, self.database_url)
        return [s.get_secret_value() for s in secrets if s.get_secret_value()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (also usable as a FastAPI dependency)."""
    return Settings()

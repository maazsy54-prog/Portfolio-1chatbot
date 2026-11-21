"""Central application settings.

All configuration lives here. Secrets are read from environment variables
(or a local, git-ignored .env file) and are never hard-coded.
"""
import re
from functools import lru_cache
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import SecretStr, field_validator
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

    # --- Retrieval (thresholds are calibrated later from eval data) ---
    retrieval_top_k: int = 20  # candidates fetched from the vector DB
    context_top_n: int = 5  # chunks passed to the LLM after reranking
    similarity_threshold: float = 0.0  # min vector score when reranking is off
    rerank_enabled: bool = True
    rerank_threshold: float = 0.0  # min rerank score when reranking is on

    # --- Generation ---
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    max_context_chars: int = 12000  # cap on excerpt text sent to the LLM

    # --- Limits ---
    max_upload_mb: int = 10
    max_pdf_pages: int = 300
    max_question_chars: int = 1000
    rate_limit: str = "30/minute"

    # --- External call resilience ---
    http_timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5  # initial backoff; grows exponentially
    embed_batch_size: int = 96  # Cohere's maximum texts per embed call
    embed_concurrency: int = 4  # parallel embed requests
    upsert_batch_size: int = 100  # vectors per Pinecone upsert request
    index_visibility_timeout_seconds: float = 15.0  # wait for fresh vectors
    index_poll_interval_seconds: float = 0.5  # how often to re-check visibility

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: object) -> object:
        """Convert a standard Postgres URL into the asyncpg form.

        - postgres:// or postgresql://  ->  postgresql+asyncpg://
        - sslmode=...                   ->  ssl=...  (asyncpg's name for it)
        - channel_binding=...           ->  dropped  (not supported by asyncpg)
        """
        if not isinstance(value, str) or not value:
            return value
        value = re.sub(r"^postgres(ql)?://", "postgresql+asyncpg://", value)
        parts = urlsplit(value)
        query = [(k, v) for k, v in parse_qsl(parts.query) if k != "channel_binding"]
        query = [("ssl", v) if k == "sslmode" else (k, v) for k, v in query]
        return urlunsplit(parts._replace(query=urlencode(query)))

    def secret_values(self) -> list[str]:
        """Return all non-empty secret values (used by the log redactor)."""
        secrets = (self.cohere_api_key, self.pinecone_api_key, self.database_url)
        return [s.get_secret_value() for s in secrets if s.get_secret_value()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (also usable as a FastAPI dependency)."""
    return Settings()

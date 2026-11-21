"""Pydantic request and response models (these drive the OpenAPI docs)."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings


class HealthResponse(BaseModel):
    """Liveness information about the running service."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok", "app": "Production RAG Service", "version": "0.1.0",
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


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: object | None = None


class ErrorResponse(BaseModel):
    """Every error in the API has this shape."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": "duplicate_document",
                    "message": "This document has already been uploaded.",
                    "request_id": "9f1c2d3e4a5b",
                    "details": {"document_id": "0b7e…", "filename": "policy.pdf"},
                }
            }
        }
    )

    error: ErrorBody


class DocumentOut(BaseModel):
    """A stored document."""

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "0b7e6a3c-1d2f-4c8e-9a10-5b6c7d8e9f01", "filename": "nimbus_faq.pdf",
                "file_type": "pdf", "size_bytes": 48213, "page_count": 4, "chunk_count": 12,
                "status": "ready", "created_at": "2026-11-21T12:00:00Z",
            }
        },
    )

    id: str
    filename: str
    file_type: str
    size_bytes: int
    page_count: int | None
    chunk_count: int
    status: str
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentOut]
    total: int


class AskRequest(BaseModel):
    """A question about the uploaded documents."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"question": "How long does standard shipping take?", "document_ids": None}
        }
    )

    question: str = Field(description="The question, in plain language.")
    document_ids: list[str] | None = Field(
        default=None, description="Optionally restrict the search to these documents."
    )
    use_rerank: bool | None = Field(
        default=None, description="Override the server's reranking default (for comparisons)."
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty.")
        limit = get_settings().max_question_chars
        if len(value) > limit:
            raise ValueError(f"Question is too long (maximum {limit} characters).")
        return value


class ChunkOut(BaseModel):
    """A retrieved passage, shown so the user can verify the answer."""

    id: str
    document_id: str
    filename: str
    page: int | None
    section: str | None
    position: int
    text: str
    score: float = Field(description="Vector similarity (cosine).")
    rerank_score: float | None = None
    cited: bool = Field(description="True if the answer cites this chunk.")


class CitationOut(BaseModel):
    number: int
    chunk_id: str
    document_id: str
    filename: str
    page: int | None
    section: str | None


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    llm_calls: int


class AskResponse(BaseModel):
    """The answer, its citations and the evidence behind it."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "answer": "Standard shipping takes 5 to 7 business days [1].",
                "refused": False, "refusal_reason": None,
                "citations": [{"number": 1, "chunk_id": "0b7e…:0", "document_id": "0b7e…",
                               "filename": "nimbus_faq.pdf", "page": 1, "section": None}],
                "retrieved_chunks": [{"id": "0b7e…:0", "document_id": "0b7e…",
                                      "filename": "nimbus_faq.pdf", "page": 1, "section": None,
                                      "position": 0, "text": "Q1. How long does standard shipping take? …",
                                      "score": 0.71, "rerank_score": 0.93, "cited": True}],
                "chunks_fetched": 20, "chunks_returned": 5, "reranked": True,
                "usage": {"input_tokens": 912, "output_tokens": 41, "llm_calls": 1},
                "request_id": "9f1c2d3e4a5b", "latency_ms": 1840.2,
            }
        }
    )

    answer: str
    refused: bool
    refusal_reason: str | None
    citations: list[CitationOut]
    retrieved_chunks: list[ChunkOut]
    chunks_fetched: int = Field(description="Candidates returned by the vector database.")
    chunks_returned: int = Field(description="Chunks kept after reranking and thresholds.")
    reranked: bool
    usage: UsageOut
    request_id: str
    latency_ms: float

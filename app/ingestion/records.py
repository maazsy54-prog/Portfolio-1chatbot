"""Turn chunks + embeddings into vector records with citation metadata."""
from app.clients.pinecone_client import VectorRecord
from app.ingestion.chunking import Chunk


def make_vector_id(document_id: str, position: int) -> str:
    """Deterministic id: lets us delete a document's vectors without a search."""
    return f"{document_id}:{position}"


def build_records(
    document_id: str, filename: str, chunks: list[Chunk], vectors: list[list[float]]
) -> list[VectorRecord]:
    """Pair each chunk with its vector. Metadata supports citation and filtering.

    The chunk text is stored in the vector metadata so retrieval needs one
    round trip. Chunks are small, far below Pinecone's 40 KB metadata limit.
    """
    if len(chunks) != len(vectors):
        raise ValueError("chunks and vectors must have the same length")
    return [
        VectorRecord(
            id=make_vector_id(document_id, chunk.position),
            values=vector,
            metadata={
                "document_id": document_id,
                "filename": filename,
                "position": chunk.position,
                "page": chunk.page,
                "section": chunk.section,
                "token_count": chunk.token_count,
                "text": chunk.text,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

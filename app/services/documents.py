"""Document lifecycle: ingest, list, fetch, delete.

Consistency rules:
* A document is `processing` until every chunk is embedded, stored in Pinecone
  AND recorded in Postgres; only then is it `ready` and searchable.
* If anything fails midway, vectors already written are deleted and the row is
  removed (or marked `failed` if even cleanup could not finish).
* Deleting removes the vectors first; retrieval additionally ignores any vector
  whose document is not `ready`, so deleted content can never reach an answer.
"""
import asyncio
import logging

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.components import Components
from app.config import Settings
from app.db.models import (
    PUBLIC_OWNER, STATUS_DELETING, STATUS_FAILED, STATUS_PROCESSING, STATUS_READY,
    Chunk, Document,
)
from app.errors import BadRequestError, DuplicateDocumentError, NotFoundError
from app.ingestion.chunking import chunk_document
from app.ingestion.extract import extract_text
from app.ingestion.records import build_records
from app.ingestion.validate import validate_upload

logger = logging.getLogger(__name__)


def _duplicate_error(existing: Document) -> DuplicateDocumentError:
    return DuplicateDocumentError(
        "This document has already been uploaded.",
        details={"document_id": existing.id, "filename": existing.filename},
    )


async def ingest_document(
    session: AsyncSession,
    *,
    filename: str,
    data: bytes,
    components: Components,
    settings: Settings,
    owner_id: str = PUBLIC_OWNER,
    namespace: str | None = None,
) -> Document:
    """Validate, extract, chunk, embed and store one uploaded file.

    Raises:
        BadRequestError / PayloadTooLargeError / UnsupportedMediaTypeError: bad file.
        DuplicateDocumentError: identical content already uploaded.
        ServiceUnavailableError: embedding service or vector DB unavailable.
    """
    upload = validate_upload(filename, data, settings.max_upload_mb * 1024 * 1024)

    existing = (
        await session.execute(
            select(Document).where(Document.owner_id == owner_id, Document.sha256 == upload.sha256)
        )
    ).scalar_one_or_none()
    if existing:
        raise _duplicate_error(existing)

    document = Document(
        owner_id=owner_id, filename=upload.filename, sha256=upload.sha256,
        file_type=upload.file_type, size_bytes=upload.size_bytes, status=STATUS_PROCESSING,
    )
    session.add(document)
    try:
        await session.commit()  # the unique constraint settles simultaneous identical uploads
    except IntegrityError:
        await session.rollback()
        winner = (
            await session.execute(
                select(Document).where(Document.owner_id == owner_id, Document.sha256 == upload.sha256)
            )
        ).scalar_one_or_none()
        raise _duplicate_error(winner) if winner else DuplicateDocumentError(
            "This document has already been uploaded."
        )

    vector_ids: list[str] = []
    try:
        # CPU-bound work runs in threads so other requests are not blocked.
        extracted = await asyncio.to_thread(
            extract_text, upload.file_type, data, max_pages=settings.max_pdf_pages
        )
        chunks = await asyncio.to_thread(
            chunk_document, extracted,
            size=settings.chunk_size_tokens, overlap=settings.chunk_overlap_tokens,
        )
        if not chunks:
            raise BadRequestError("No text could be extracted from the document.")

        vectors = await components.embedder.embed_documents([c.text for c in chunks])
        records = build_records(document.id, document.filename, chunks, vectors)
        vector_ids = [r.id for r in records]
        await components.store.upsert(records, namespace=namespace)
        if not await components.store.wait_until_visible(records[-1], namespace=namespace):
            logger.warning("Vectors for %s not yet visible; marking ready anyway", document.id)

        session.add_all(
            Chunk(
                document_id=document.id, position=c.position, page=c.page, section=c.section,
                token_count=c.token_count, text=c.text, vector_id=r.id,
            )
            for c, r in zip(chunks, records)
        )
        document.status = STATUS_READY
        document.chunk_count = len(chunks)
        document.page_count = extracted.page_count
        await session.commit()
    except Exception:
        await _cleanup_failed(session, document.id, vector_ids, components, namespace)
        raise
    return document


async def _cleanup_failed(
    session: AsyncSession, document_id: str, vector_ids: list[str],
    components: Components, namespace: str | None,
) -> None:
    """Undo a half-finished ingestion (best effort; never raises)."""
    try:
        await session.rollback()
        if vector_ids:
            await components.store.delete(vector_ids, namespace=namespace)
        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        await session.execute(delete(Document).where(Document.id == document_id))
        await session.commit()
    except Exception:
        logger.exception("Cleanup after failed ingestion of %s did not complete", document_id)
        try:
            await session.rollback()
            doc = await session.get(Document, document_id)
            if doc:
                doc.status = STATUS_FAILED  # leaves a trace for manual cleanup
                await session.commit()
        except Exception:
            logger.exception("Could not mark %s as failed", document_id)


async def list_documents(session: AsyncSession, owner_id: str = PUBLIC_OWNER) -> list[Document]:
    """Documents visible to the user, newest first."""
    result = await session.execute(
        select(Document)
        .where(Document.owner_id == owner_id, Document.status.in_([STATUS_READY, STATUS_PROCESSING]))
        .order_by(Document.created_at.desc())
    )
    return list(result.scalars())


async def get_document(session: AsyncSession, document_id: str, owner_id: str = PUBLIC_OWNER) -> Document:
    """Fetch one document or raise NotFoundError."""
    doc = await session.get(Document, document_id)
    if not doc or doc.owner_id != owner_id or doc.status in (STATUS_DELETING, STATUS_FAILED):
        raise NotFoundError("Document not found.")
    return doc


async def delete_document(
    session: AsyncSession, document_id: str, *, components: Components,
    owner_id: str = PUBLIC_OWNER, namespace: str | None = None,
) -> None:
    """Delete a document and all of its vectors.

    Order: mark `deleting` (hides it from search immediately) -> delete vectors
    -> delete rows. If the vector delete fails the row stays `deleting`, so the
    request can simply be retried.
    """
    doc = await session.get(Document, document_id)
    if not doc or doc.owner_id != owner_id:
        raise NotFoundError("Document not found.")
    doc.status = STATUS_DELETING
    await session.commit()

    vector_ids = list(
        (await session.execute(select(Chunk.vector_id).where(Chunk.document_id == document_id))).scalars()
    )
    await components.store.delete(vector_ids, namespace=namespace)
    await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
    await session.execute(delete(Document).where(Document.id == document_id))
    await session.commit()

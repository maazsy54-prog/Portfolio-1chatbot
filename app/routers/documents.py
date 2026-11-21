"""Document routes: upload, list, fetch, delete. No business logic here."""
from typing import Annotated

from fastapi import APIRouter, File, Response, UploadFile, status

from app.dependencies import ComponentsDep, SessionDep, SettingsDep
from app.schemas import DocumentListResponse, DocumentOut, ErrorResponse
from app.services import documents as service

router = APIRouter(prefix="/documents", tags=["Documents"])

_ERRORS = {
    400: {"model": ErrorResponse, "description": "Empty file, corrupt file, or no extractable text."},
    404: {"model": ErrorResponse, "description": "Document does not exist."},
    409: {"model": ErrorResponse, "description": "The same document was already uploaded."},
    413: {"model": ErrorResponse, "description": "File is larger than the upload limit."},
    415: {"model": ErrorResponse, "description": "Unsupported (or mislabelled) file type."},
    422: {"model": ErrorResponse, "description": "Invalid request data."},
    503: {"model": ErrorResponse, "description": "Embedding service or vector database unavailable."},
}


@router.post(
    "", status_code=status.HTTP_201_CREATED, response_model=DocumentOut,
    summary="Upload and ingest a document",
    description="Accepts a PDF or DOCX. The file is validated, text is extracted, split into "
    "chunks, embedded and stored. Returns 201 only once the document is searchable.",
    responses={k: _ERRORS[k] for k in (400, 409, 413, 415, 422, 503)},
)
async def upload_document(
    file: Annotated[UploadFile, File(description="A .pdf or .docx file")],
    session: SessionDep, components: ComponentsDep, settings: SettingsDep,
) -> DocumentOut:
    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)  # never read more than limit + 1 byte
    document = await service.ingest_document(
        session, filename=file.filename or "document", data=data,
        components=components, settings=settings,
    )
    return DocumentOut.model_validate(document)


@router.get("", response_model=DocumentListResponse, summary="List uploaded documents")
async def list_documents(session: SessionDep) -> DocumentListResponse:
    docs = await service.list_documents(session)
    return DocumentListResponse(
        documents=[DocumentOut.model_validate(d) for d in docs], total=len(docs)
    )


@router.get(
    "/{document_id}", response_model=DocumentOut, summary="Get one document",
    responses={404: _ERRORS[404]},
)
async def get_document(document_id: str, session: SessionDep) -> DocumentOut:
    return DocumentOut.model_validate(await service.get_document(session, document_id))


@router.delete(
    "/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a document",
    description="Removes the document, its chunks and all of its vectors. Its content can "
    "never appear in answers afterwards.",
    responses={404: _ERRORS[404], 503: _ERRORS[503]},
)
async def delete_document(
    document_id: str, session: SessionDep, components: ComponentsDep
) -> Response:
    await service.delete_document(session, document_id, components=components)
    return Response(status_code=status.HTTP_204_NO_CONTENT)

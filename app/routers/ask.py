"""Question-answering route. No business logic here."""
from fastapi import APIRouter, Depends, Request

from app.dependencies import ComponentsDep, SessionDep, SettingsDep
from app.ratelimit import limit_ask
from app.schemas import AskRequest, AskResponse, ErrorResponse
from app.services.qa import answer_question

router = APIRouter(tags=["Question answering"])


@router.post(
    "/ask", response_model=AskResponse, dependencies=[Depends(limit_ask)], summary="Ask a question about your documents",
    description="Retrieves the most relevant passages and answers from them with numbered "
    "citations. If the documents do not contain the answer, the response has `refused: true` "
    "and says so; nothing is invented. The retrieved chunks and scores are always returned.",
    responses={
        404: {"model": ErrorResponse, "description": "A document in document_ids does not exist."},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded (see the Retry-After header)."},
        422: {"model": ErrorResponse, "description": "Empty or overly long question."},
        503: {"model": ErrorResponse, "description": "LLM, embedding service or vector database unavailable."},
    },
)
async def ask(
    body: AskRequest, request: Request, session: SessionDep,
    components: ComponentsDep, settings: SettingsDep,
) -> AskResponse:
    return await answer_question(
        session, question=body.question, document_ids=body.document_ids,
        use_rerank=body.use_rerank, components=components, settings=settings,
        request_id=request.state.request_id,
    )

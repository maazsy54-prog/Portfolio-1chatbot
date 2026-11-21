"""API tests for /ask: citations, refusals, deletion safety and validation."""
from tests.conftest import RETURNS_DOCX, SHIPPING_PDF, DOCX_MIME, upload

Q = "How long does standard shipping take?"


def ask(env, question: str = Q, **extra):
    return env.client.post("/ask", json={"question": question, **extra})


def test_ask_before_any_upload_refuses_without_any_api_calls(env) -> None:
    response = ask(env)
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] and body["refusal_reason"] == "no_documents"
    assert body["chunks_fetched"] == 0 and body["citations"] == []
    assert env.embedder.calls == 0 and env.llm.calls == 0


def test_answer_has_citations_that_point_at_retrieved_chunks(env) -> None:
    upload(env, "shipping.pdf", SHIPPING_PDF)
    body = ask(env).json()
    assert not body["refused"] and body["answer"]
    retrieved_ids = {c["id"] for c in body["retrieved_chunks"]}
    assert body["citations"] and all(c["chunk_id"] in retrieved_ids for c in body["citations"])
    assert body["citations"][0]["filename"] == "shipping.pdf" and body["citations"][0]["page"] == 1
    assert body["chunks_fetched"] >= 1 and body["chunks_returned"] == len(body["retrieved_chunks"])
    chunk = body["retrieved_chunks"][0]
    assert isinstance(chunk["score"], float) and chunk["cited"] is True
    assert body["usage"]["llm_calls"] == 1 and body["request_id"] and body["latency_ms"] >= 0


def test_unanswerable_question_is_refused_clearly(env) -> None:
    upload(env, "shipping.pdf", SHIPPING_PDF)
    body = ask(env, "What is the unknowable secret of the universe?").json()
    assert body["refused"] and body["refusal_reason"] == "not_in_documents"
    assert body["citations"] == []


def test_ask_after_deleting_everything_refuses(env) -> None:
    doc = upload(env, "shipping.pdf", SHIPPING_PDF).json()
    env.client.delete(f"/documents/{doc['id']}")
    body = ask(env).json()
    assert body["refused"] and body["refusal_reason"] == "no_documents"


def test_deleted_content_never_appears_even_if_vector_delete_lags(env) -> None:
    shipping = upload(env, "shipping.pdf", SHIPPING_PDF).json()
    upload(env, "returns.docx", RETURNS_DOCX, DOCX_MIME)
    env.store.ignore_deletes = True  # Pinecone still holds the deleted vectors
    assert env.client.delete(f"/documents/{shipping['id']}").status_code == 204
    assert any(r.metadata["document_id"] == shipping["id"] for r in env.store.data.values())
    body = ask(env).json()
    assert all(c["document_id"] != shipping["id"] for c in body["retrieved_chunks"])
    assert all(c["filename"] != "shipping.pdf" for c in body["citations"])


def test_document_filter_restricts_search(env) -> None:
    upload(env, "shipping.pdf", SHIPPING_PDF)
    returns = upload(env, "returns.docx", RETURNS_DOCX, DOCX_MIME).json()
    body = ask(env, document_ids=[returns["id"]]).json()
    assert {c["document_id"] for c in body["retrieved_chunks"]} == {returns["id"]}


def test_unknown_document_id_is_404(env) -> None:
    upload(env, "shipping.pdf", SHIPPING_PDF)
    assert ask(env, document_ids=["nope"]).status_code == 404


def test_rerank_can_be_disabled_per_request(env) -> None:
    upload(env, "shipping.pdf", SHIPPING_PDF)
    assert ask(env, use_rerank=False).json()["reranked"] is False
    assert ask(env, use_rerank=True).json()["reranked"] is True


def test_invalid_questions_are_422(env) -> None:
    for bad in ["", "   ", "x" * 1001]:
        response = env.client.post("/ask", json={"question": bad})
        assert response.status_code == 422, bad[:10]
        assert response.json()["error"]["code"] == "validation_error"
    assert env.client.post("/ask", json={}).status_code == 422


def test_embedding_outage_during_ask_is_503(env) -> None:
    upload(env, "shipping.pdf", SHIPPING_PDF)
    env.embedder.fail = True
    response = ask(env)
    assert response.status_code == 503 and response.json()["error"]["code"] == "service_unavailable"


def test_question_logged_for_dashboards(env) -> None:
    upload(env, "shipping.pdf", SHIPPING_PDF)
    body = ask(env).json()
    import asyncio
    from sqlalchemy import select
    from app.db.models import QueryLog

    async def fetch():
        async with env.client.app.state.components.session_factory() as s:
            return (await s.execute(select(QueryLog))).scalars().all()

    rows = env.client.portal.call(fetch)
    assert [r.request_id for r in rows] == [body["request_id"]]
    assert rows[0].input_tokens == 300

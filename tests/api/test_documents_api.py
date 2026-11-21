"""API tests for upload, list, get and delete, including every failure path."""
from concurrent.futures import ThreadPoolExecutor

from tests.conftest import (
    DOCX_MIME, RETURNS_DOCX, SHIPPING_PDF, make_pdf, upload,
)


def test_upload_pdf_returns_201_and_document_is_listed(env) -> None:
    response = upload(env, "shipping.pdf", SHIPPING_PDF)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ready" and body["file_type"] == "pdf"
    assert body["chunk_count"] >= 1 and body["page_count"] == 1
    listing = env.client.get("/documents").json()
    assert listing["total"] == 1 and listing["documents"][0]["id"] == body["id"]
    assert len(env.store.data) == body["chunk_count"]
    assert env.client.get(f"/documents/{body['id']}").json()["filename"] == "shipping.pdf"


def test_upload_docx_returns_201(env) -> None:
    response = upload(env, "returns.docx", RETURNS_DOCX, DOCX_MIME)
    assert response.status_code == 201 and response.json()["file_type"] == "docx"


def test_stored_vectors_carry_citation_metadata(env) -> None:
    upload(env, "shipping.pdf", SHIPPING_PDF)
    record = next(iter(env.store.data.values()))
    assert record.metadata["filename"] == "shipping.pdf" and record.metadata["page"] == 1
    assert record.metadata["text"] and record.metadata["position"] == 0


def test_duplicate_upload_is_409_and_not_stored_twice(env) -> None:
    first = upload(env, "shipping.pdf", SHIPPING_PDF).json()
    vectors_before = len(env.store.data)
    second = upload(env, "renamed-copy.pdf", SHIPPING_PDF)
    assert second.status_code == 409
    error = second.json()["error"]
    assert error["code"] == "duplicate_document" and error["details"]["document_id"] == first["id"]
    assert env.client.get("/documents").json()["total"] == 1
    assert len(env.store.data) == vectors_before


def test_same_filename_different_content_is_allowed(env) -> None:
    assert upload(env, "doc.pdf", SHIPPING_PDF).status_code == 201
    other = make_pdf("A completely different document about warranty periods and repairs for tents.")
    assert upload(env, "doc.pdf", other).status_code == 201


def test_simultaneous_identical_uploads_only_one_wins(env) -> None:
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: upload(env, "same.pdf", SHIPPING_PDF), range(2)))
    codes = sorted(r.status_code for r in results)
    assert codes == [201, 409], codes
    assert env.client.get("/documents").json()["total"] == 1


def test_unsupported_extension_is_415(env) -> None:
    response = upload(env, "notes.txt", b"hello world", "text/plain")
    assert response.status_code == 415 and response.json()["error"]["code"] == "unsupported_media_type"


def test_renamed_executable_is_415(env) -> None:
    assert upload(env, "evil.pdf", b"MZ\x90\x00 not really a pdf").status_code == 415


def test_empty_file_is_400(env) -> None:
    assert upload(env, "empty.pdf", b"").status_code == 400


def test_scanned_pdf_is_400_and_nothing_is_stored(env) -> None:
    response = upload(env, "scan.pdf", make_pdf("", ""))
    assert response.status_code == 400 and "extractable text" in response.json()["error"]["message"]
    assert env.client.get("/documents").json()["total"] == 0 and not env.store.data


def test_corrupt_pdf_is_400(env) -> None:
    assert upload(env, "bad.pdf", b"%PDF-1.4 garbage that is not a pdf").status_code == 400


def test_too_large_file_is_413(env) -> None:
    response = upload(env, "big.pdf", b"%PDF-1.4\n" + b"x" * (1024 * 1024 + 10))
    assert response.status_code == 413


def test_missing_file_field_is_422_with_consistent_shape(env) -> None:
    response = env.client.post("/documents")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_path_traversal_filename_is_sanitized(env) -> None:
    body = upload(env, "../../etc/shipping.pdf", SHIPPING_PDF).json()
    assert body["filename"] == "shipping.pdf"


def test_get_unknown_document_is_404(env) -> None:
    assert env.client.get("/documents/does-not-exist").status_code == 404


def test_delete_removes_document_and_vectors(env) -> None:
    doc = upload(env, "shipping.pdf", SHIPPING_PDF).json()
    assert env.store.data
    assert env.client.delete(f"/documents/{doc['id']}").status_code == 204
    assert not env.store.data
    assert env.client.get("/documents").json()["total"] == 0
    assert env.client.get(f"/documents/{doc['id']}").status_code == 404
    assert env.client.delete(f"/documents/{doc['id']}").status_code == 404


def test_deleted_document_can_be_uploaded_again(env) -> None:
    doc = upload(env, "shipping.pdf", SHIPPING_PDF).json()
    env.client.delete(f"/documents/{doc['id']}")
    assert upload(env, "shipping.pdf", SHIPPING_PDF).status_code == 201


def test_embedding_outage_is_503_and_leaves_nothing_behind(env) -> None:
    env.embedder.fail = True
    response = upload(env, "shipping.pdf", SHIPPING_PDF)
    assert response.status_code == 503 and response.json()["error"]["code"] == "service_unavailable"
    assert env.client.get("/documents").json()["total"] == 0 and not env.store.data
    env.embedder.fail = False
    assert upload(env, "shipping.pdf", SHIPPING_PDF).status_code == 201  # retry works


def test_failure_after_upsert_rolls_back_vectors_and_row(env) -> None:
    env.store.fail_visible = True  # vectors are written, then the pipeline fails
    response = upload(env, "shipping.pdf", SHIPPING_PDF)
    assert response.status_code == 503
    assert not env.store.data, "orphan vectors were left in the vector database"
    assert env.client.get("/documents").json()["total"] == 0

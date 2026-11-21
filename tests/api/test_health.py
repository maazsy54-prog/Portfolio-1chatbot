"""Smoke tests for the scaffold: health route, error shape and request IDs."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert set(body["configured"]) == {"cohere", "pinecone", "database"}


def test_unknown_route_uses_consistent_error_shape() -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "not_found"
    assert error["request_id"]


def test_request_id_is_echoed_when_valid() -> None:
    response = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"


def test_unsafe_request_id_is_replaced() -> None:
    response = client.get("/health", headers={"X-Request-ID": "bad id!\t"})
    assert response.headers["X-Request-ID"] != "bad id!\t"

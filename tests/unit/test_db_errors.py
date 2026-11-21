"""A database outage must surface as a clean 503, not a stack trace."""
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.errors import register_exception_handlers
from app.logging_conf import RequestContextMiddleware


def test_operational_error_maps_to_503() -> None:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    response = TestClient(app).get("/boom")
    assert response.status_code == 503
    body = response.json()["error"]
    assert body["code"] == "service_unavailable" and "connection refused" not in body["message"]

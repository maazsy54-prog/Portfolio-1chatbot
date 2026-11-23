"""The web UI is served by the API and exposes everything the spec requires."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_redirects_to_ui() -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307) and response.headers["location"] == "/app/"


def test_ui_page_is_served_with_required_sections() -> None:
    html = client.get("/app/").text
    assert client.get("/app/").headers["content-type"].startswith("text/html")
    for needle in ("Add documents", "Your documents", 'id="askForm"', 'id="thread"', "fileInput"):
        assert needle in html, needle


def test_ui_assets_are_served() -> None:
    js = client.get("/app/app.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert "Retrieved chunks" in js.text and "fetched from the vector database" in js.text
    assert client.get("/app/styles.css").status_code == 200


def test_static_mount_does_not_expose_files_outside_frontend() -> None:
    for path in ("/app/../.env", "/app/%2e%2e/.env", "/app/..%2fapp%2fconfig.py"):
        assert client.get(path).status_code in (400, 404), path


def test_unknown_api_route_still_returns_json_error() -> None:
    response = client.get("/does-not-exist")
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"

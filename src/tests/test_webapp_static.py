from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from getwork.webapp import api, demo, explain

pytestmark = pytest.mark.usefixtures("offline")

INDEX_HTML = "<!doctype html><title>Built GetWork</title><div id=root></div>"
PLACEHOLDER_COMMAND = "npm --prefix src/web run build"


def make_client(static_dir: Path | None) -> TestClient:
    app = api.create_app(demo.DemoEngine(), explain.TemplateExplainer(), static_dir=static_dir)
    return TestClient(app)


def assert_api_still_works(client: TestClient) -> None:
    assert client.get("/api/health").status_code == 200
    unknown = client.get("/api/nope")
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "not_found"


def test_serves_built_frontend(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("console.log('ok');", encoding="utf-8")
    client = make_client(tmp_path)
    index = client.get("/")
    assert index.status_code == 200
    assert index.text == INDEX_HTML
    asset = client.get("/assets/app.js")
    assert asset.status_code == 200
    assert "javascript" in asset.headers["content-type"]
    assert asset.text == "console.log('ok');"
    assert_api_still_works(client)


def test_placeholder_without_build(tmp_path: Path) -> None:
    empty_dir = tmp_path / "dist"
    empty_dir.mkdir()
    for static_dir in (None, empty_dir):
        client = make_client(static_dir)
        page = client.get("/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "Frontend not built" in page.text
        assert PLACEHOLDER_COMMAND in page.text
        assert "/docs" in page.text
        assert "http://" not in page.text and "https://" not in page.text
        assert_api_still_works(client)

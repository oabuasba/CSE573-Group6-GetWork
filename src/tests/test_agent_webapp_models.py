import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from getwork.webapp import api, demo, explain, models

pytestmark = pytest.mark.usefixtures("offline")

REPO = Path(__file__).resolve().parents[2]
CONTRACT = REPO / "src" / "web" / "src" / "fixtures" / "contract.json"


def test_explanation_has_no_claim_fields() -> None:
    assert set(models.Explanation.model_fields) == {"summary", "bullets", "evidence", "explainer"}


def test_old_explanation_payload_still_valid() -> None:
    fixture = json.loads(CONTRACT.read_text())
    parsed = models.Explanation.model_validate(fixture["explain"])
    assert parsed.explainer.backend is None and parsed.explainer.prompt_version is None


def test_explainer_info_extra_fields() -> None:
    info = models.ExplainerInfo(kind="agent", model="m", backend="voyager", prompt_version="agent_v2")
    assert (info.backend, info.prompt_version) == ("voyager", "agent_v2")
    again = models.ExplainerInfo.model_validate(info.model_dump())
    assert again == info


def test_webapp_models_do_not_import_agent() -> None:
    code = "import sys, getwork.webapp.models; sys.exit(any(m.startswith('getwork.agent') for m in sys.modules))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, check=False)
    assert done.returncode == 0
    assert "getwork.agent" not in (REPO / "src/getwork/webapp/models.py").read_text()


def test_webapp_error_codes_unchanged() -> None:
    assert models.ERROR_CODES == {
        "invalid_request": 422,
        "empty_query": 400,
        "text_too_long": 400,
        "side_mismatch": 400,
        "not_found": 404,
        "embedding_mismatch": 409,
        "structuring_failed": 502,
        "embedding_failed": 502,
        "capability_unavailable": 503,
        "internal": 500,
    }


def test_pyproject_agent_extra_and_package_data() -> None:
    data = tomllib.loads((REPO / "pyproject.toml").read_text())
    assert "langgraph" in data["project"]["optional-dependencies"]["agent"]
    assert "langgraph" in data["dependency-groups"]["dev"]
    assert data["tool"]["setuptools"]["package-data"]["getwork.agent"] == ["prompts/*.md"]


def test_langgraph_importable() -> None:
    import langgraph.graph  # noqa: F401


def test_explainer_info_omits_unset_fields() -> None:
    assert models.ExplainerInfo(kind="agent", model="m").model_dump() == {"kind": "agent", "model": "m"}
    assert models.ExplainerInfo(kind="template").model_dump() == {"kind": "template", "model": None}
    full = models.ExplainerInfo(kind="agent", model="m", backend="voyager", prompt_version="agent_v2")
    assert full.model_dump() == {
        "kind": "agent",
        "model": "m",
        "backend": "voyager",
        "prompt_version": "agent_v2",
    }
    health = TestClient(api.create_app(demo.DemoEngine(), explain.TemplateExplainer())).get("/api/health").json()
    assert health["explainer"] == {"kind": "template", "model": None}

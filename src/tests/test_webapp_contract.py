import json
from pathlib import Path

import pytest

from getwork.webapp import models

pytestmark = pytest.mark.usefixtures("offline")

CONTRACT = Path(__file__).resolve().parents[1] / "web" / "src" / "fixtures" / "contract.json"
SECTIONS = {
    "health": models.HealthResponse,
    "examples": models.ExamplesResponse,
    "match": models.MatchResponse,
    "explain": models.Explanation,
    "error": models.ErrorEnvelope,
}


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_contract_fixture_validates() -> None:
    data = load_contract()
    assert set(data) == set(SECTIONS)
    for key, model in SECTIONS.items():
        model.model_validate(data[key])


def test_contract_covers_all_evidence_kinds() -> None:
    match = models.MatchResponse.model_validate(load_contract()["match"])
    assert len(match.matches) >= 3
    assert any(m.evidence.shared and m.evidence.related and m.evidence.gaps for m in match.matches)
    health = models.HealthResponse.model_validate(load_contract()["health"])
    assert health.mode == "demo" and health.versions.demo
    assert {"structure", "embed"} <= set(health.capabilities)
    assert health.levels

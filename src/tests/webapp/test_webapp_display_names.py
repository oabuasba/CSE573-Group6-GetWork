import pytest
from fastapi.testclient import TestClient

from getwork.agent.models import ChatMatchRef, ChatRequest
from getwork.webapp import api, demo, explain, models, pseudonym

pytestmark = pytest.mark.usefixtures("offline")

SYSTEMS = ("B1+", "B2", "Full")
MATCH_KEYS = {"rank", "id", "score", "document", "evidence", "display_name"}


def make_client() -> TestClient:
    return TestClient(api.create_app(demo.DemoEngine(), explain.TemplateExplainer()))


def candidate_body(**extra: object) -> dict:
    query = {"kind": "corpus", "id": "j-1"}
    return {"direction": "posting_to_resume", "query": query, "k": 6, **extra}


def test_candidate_matches_have_names() -> None:
    response = make_client().post("/api/match", json=candidate_body())
    assert response.status_code == 200
    body = models.MatchResponse.model_validate(response.json())
    assert body.matches
    for match in body.matches:
        assert match.display_name
        assert match.display_name == pseudonym.display_name(match.id)


def test_job_matches_have_no_name(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(doc_id: str | None) -> str | None:
        raise AssertionError("display_name must not be called for job matches")

    monkeypatch.setattr(pseudonym, "display_name", boom)
    body = {"direction": "resume_to_posting", "query": {"kind": "corpus", "id": "r-1"}}
    response = make_client().post("/api/match", json=body)
    assert response.status_code == 200
    matches = response.json()["matches"]
    assert matches
    assert all(match["display_name"] is None for match in matches)


def test_name_only_on_match() -> None:
    assert "display_name" in models.Match.model_fields
    others = (
        models.DocumentView,
        models.Evidence,
        models.Explanation,
        models.ExamplesResponse,
        ChatRequest,
        ChatMatchRef,
    )
    for model in others:
        assert "display_name" not in model.model_fields, model.__name__

    client = make_client()
    body = client.post("/api/match", json=candidate_body()).json()
    names = [match["display_name"] for match in body["matches"]]
    for match in body["matches"]:
        assert set(match) == MATCH_KEYS
        assert set(match["document"]) == set(models.DocumentView.model_fields)

    examples = client.get("/api/examples", params={"side": "resume"})
    query = demo.DemoEngine().get_document("j-1").model_dump(mode="json")
    explained = client.post(
        "/api/explain",
        json={"direction": "posting_to_resume", "query": query, "match_id": "r-1"},
    )
    assert examples.status_code == explained.status_code == 200
    for text in (examples.text, explained.text):
        assert "display_name" not in text
        assert not any(name in text for name in names)


@pytest.mark.parametrize("system", SYSTEMS)
def test_rankings_unaffected(system: str, monkeypatch: pytest.MonkeyPatch) -> None:
    body = candidate_body(system=system)
    expected = demo.DemoEngine().match(models.MatchRequest.model_validate(body))
    expected_pairs = [(doc.id, score) for doc, score in expected.matches]
    client = make_client()

    real = client.post("/api/match", json=body).json()
    monkeypatch.setattr(pseudonym, "display_name", lambda doc_id: "Constant Name")
    patched = client.post("/api/match", json=body).json()

    assert [(m["id"], m["score"]) for m in real["matches"]] == expected_pairs
    for response in (real, patched):
        for match in response["matches"]:
            match.pop("display_name")
    assert real == patched

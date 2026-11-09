import logging

import pytest
from fastapi.testclient import TestClient

from getwork.webapp import api, demo, engine, explain, models

pytestmark = pytest.mark.usefixtures("offline")

SYSTEMS = ("B1+", "B2", "Full")
CORPUS_BY_DIRECTION = {"resume_to_posting": "r-1", "posting_to_resume": "j-1"}


def make_client(**kwargs: object) -> TestClient:
    return TestClient(api.create_app(demo.DemoEngine(), explain.TemplateExplainer(), **kwargs))


def corpus_body(direction: str = "resume_to_posting", **extra: object) -> dict:
    query = {"kind": "corpus", "id": CORPUS_BY_DIRECTION[direction]}
    return {"direction": direction, "query": query, **extra}


def error_of(response) -> dict:
    body = models.ErrorEnvelope.model_validate(response.json())
    return body.error.model_dump()


def test_health() -> None:
    response = make_client().get("/api/health")
    assert response.status_code == 200
    health = models.HealthResponse.model_validate(response.json())
    assert health.status == "ok"
    assert health.explainer == models.ExplainerInfo(kind="template", model=None)
    assert health.mode == "demo"


def test_examples() -> None:
    client = make_client()
    default = client.get("/api/examples", params={"side": "resume"})
    assert default.status_code == 200
    assert [i["id"] for i in default.json()["items"]] == [f"r-{n}" for n in range(1, 7)]
    postings = client.get("/api/examples", params={"side": "posting", "limit": 2}).json()
    assert [i["id"] for i in postings["items"]] == ["j-1", "j-2"]
    models.ExamplesResponse.model_validate(postings)


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"side": "foo"}, "side"),
        ({"side": "resume", "limit": 0}, "limit"),
        ({"side": "resume", "limit": 21}, "limit"),
    ],
)
def test_examples_validation(params: dict, field: str) -> None:
    response = make_client().get("/api/examples", params=params)
    assert response.status_code == 422
    error = error_of(response)
    assert error["code"] == "invalid_request"
    assert error["field"] == field


@pytest.mark.parametrize("direction", list(CORPUS_BY_DIRECTION))
@pytest.mark.parametrize("system", SYSTEMS)
def test_match_both_directions_all_systems(direction: str, system: str) -> None:
    body = corpus_body(direction, system=system, k=4)
    response = make_client().post("/api/match", json=body)
    assert response.status_code == 200
    result = models.MatchResponse.model_validate(response.json())
    assert (result.direction, result.system, result.query_kind) == (direction, system, "corpus")
    assert (result.k, result.k_effective) == (4, 4)
    assert [m.rank for m in result.matches] == [1, 2, 3, 4]
    assert result.provenance.demo is True
    assert isinstance(result.notes, list)


def test_match_evidence_matches_builder() -> None:
    response = make_client().post("/api/match", json=corpus_body(k=50))
    result = models.MatchResponse.model_validate(response.json())
    assert result.k == 50 and result.k_effective == 6
    for item in result.matches:
        expected = explain.build_evidence("resume_to_posting", result.query, item.document)
        assert item.evidence == expected
        assert item.id == item.document.id


class ExplodingEngine(demo.DemoEngine):
    def match(self, request: models.MatchRequest) -> engine.MatchResult:
        raise RuntimeError("secret-key-123 leaked")


def test_error_envelopes() -> None:
    client = make_client()
    bad_body = client.post("/api/match", json={"direction": "nope"})
    assert bad_body.status_code == 422 and error_of(bad_body)["code"] == "invalid_request"
    assert error_of(bad_body)["field"]
    unknown_field = client.post("/api/match", json=corpus_body(extra=1))
    assert unknown_field.status_code == 422 and error_of(unknown_field)["field"] == "extra"
    zero_k = client.post("/api/match", json=corpus_body(k=0))
    assert zero_k.status_code == 422 and error_of(zero_k)["field"] == "k"

    missing = {"direction": "resume_to_posting", "query": {"kind": "corpus", "id": "r-99"}}
    not_found = client.post("/api/match", json=missing)
    assert not_found.status_code == 404 and error_of(not_found)["code"] == "not_found"

    form = {"kind": "form", "title": "", "skills": []}
    empty = client.post("/api/match", json={"direction": "resume_to_posting", "query": form})
    assert empty.status_code == 400 and error_of(empty)["code"] == "empty_query"

    wrong = {"direction": "resume_to_posting", "query": {"kind": "corpus", "id": "j-1"}}
    side = client.post("/api/match", json=wrong)
    assert side.status_code == 400 and error_of(side)["code"] == "side_mismatch"

    text = {"direction": "resume_to_posting", "query": {"kind": "text", "text": "hello"}}
    unavailable = client.post("/api/match", json=text)
    assert unavailable.status_code == 503
    assert error_of(unavailable)["code"] == "capability_unavailable"

    unknown = client.get("/api/nope")
    assert unknown.status_code == 404
    assert unknown.headers["content-type"].startswith("application/json")
    assert error_of(unknown)["code"] == "not_found"


def test_internal_error_is_generic_and_redacted(caplog: pytest.LogCaptureFixture) -> None:
    app = api.create_app(ExplodingEngine(), explain.TemplateExplainer(), secrets=["secret-key-123"])
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger="getwork.webapp"):
        response = client.post("/api/match", json=corpus_body())
    assert response.status_code == 500
    assert error_of(response)["code"] == "internal"
    assert error_of(response)["message"] == "Something went wrong on the server"
    assert "secret-key-123" not in response.text
    assert caplog.records
    for record in caplog.records:
        assert "secret-key-123" not in record.getMessage()
        assert "secret-key-123" not in (record.exc_text or "")
    assert "RuntimeError" in caplog.text


def explain_body(client: TestClient, match_id: str | None = None) -> dict:
    result = client.post("/api/match", json=corpus_body(k=3)).json()
    return {
        "direction": "resume_to_posting",
        "query": result["query"],
        "match_id": match_id or result["matches"][0]["id"],
    }


def test_explain() -> None:
    client = make_client()
    response = client.post("/api/explain", json=explain_body(client))
    assert response.status_code == 200
    result = models.Explanation.model_validate(response.json())
    assert result.explainer.kind == "template"
    assert result.summary.startswith("2 shared skills, 1 related group")
    assert result.bullets[0] == "Shared skills: JavaScript, CSS."


def test_explain_errors() -> None:
    client = make_client()
    unknown = client.post("/api/explain", json=explain_body(client, "j-99"))
    assert unknown.status_code == 404 and error_of(unknown)["code"] == "not_found"
    same_side = client.post("/api/explain", json=explain_body(client, "r-2"))
    assert same_side.status_code == 400 and error_of(same_side)["code"] == "side_mismatch"
    too_many = explain_body(client)
    too_many["query"]["skills"] = too_many["query"]["skills"] * 70
    assert len(too_many["query"]["skills"]) > 200
    oversized = client.post("/api/explain", json=too_many)
    assert oversized.status_code == 422 and error_of(oversized)["code"] == "invalid_request"


class FakeExplainer:
    info = models.ExplainerInfo(kind="agent", model="fake-model")

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, str | None]] = []

    def explain(
        self, direction: models.Direction, query: models.DocumentView, match: models.DocumentView
    ) -> models.Explanation:
        self.calls.append((direction, query.id, match.id))
        return models.Explanation(
            summary="fake summary",
            bullets=[],
            evidence=explain.build_evidence(direction, query, match),
            explainer=self.info,
        )


def test_explain_uses_injected_explainer() -> None:
    fake = FakeExplainer()
    client = TestClient(api.create_app(demo.DemoEngine(), fake))
    response = client.post("/api/explain", json=explain_body(client, "j-2"))
    assert response.status_code == 200
    body = models.Explanation.model_validate(response.json())
    assert body.summary == "fake summary"
    assert body.explainer == models.ExplainerInfo(kind="agent", model="fake-model")
    assert fake.calls == [("resume_to_posting", "r-1", "j-2")]
    health = client.get("/api/health").json()
    assert health["explainer"] == {"kind": "agent", "model": "fake-model"}


def test_no_cors() -> None:
    app = api.create_app(demo.DemoEngine(), explain.TemplateExplainer())
    client = TestClient(app)
    response = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert response.status_code == 200
    assert not any(k.lower().startswith("access-control") for k in response.headers)
    assert all("cors" not in m.cls.__name__.lower() for m in app.user_middleware)


def test_wrong_method_is_405_envelope() -> None:
    response = make_client().get("/api/match")
    assert response.status_code == 405
    assert error_of(response)["code"] == "invalid_request"
    assert "POST" in response.headers["allow"]


def test_malformed_json_has_no_field() -> None:
    response = make_client().post(
        "/api/match", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert error_of(response)["code"] == "invalid_request"
    assert error_of(response)["field"] is None


def test_explain_rejects_query_side_that_disagrees_with_direction() -> None:
    client = make_client()
    body = explain_body(client)
    body["direction"] = "posting_to_resume"
    response = client.post("/api/explain", json=body)
    assert response.status_code == 400
    assert error_of(response)["code"] == "invalid_request"
    assert error_of(response)["field"] == "query"

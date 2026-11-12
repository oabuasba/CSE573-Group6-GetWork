import logging

import pytest
from agent_helpers import SECRET, chat_json, explain_json, good_answer, make_agent, make_client
from fastapi.testclient import TestClient

from getwork.agent.api import register_chat_routes
from getwork.agent.failure import FAILURE_COPY, AgentFailure
from getwork.agent.models import ChatResponse
from getwork.webapp import api, demo, models

pytestmark = pytest.mark.usefixtures("offline")

REASONS = ["timeout", "budget_exhausted", "llm_unavailable", "model_rejected", "invalid_reply", "unverified"]
SECRET_MESSAGE = "ZEBRA-QUESTION-7"


class StubService:
    """Chat service and explainer in one: raises what it is told to."""

    info = models.ExplainerInfo(kind="agent", model="stub", backend="voyager", prompt_version="agent_v1")

    def __init__(self, error: Exception) -> None:
        self.error = error

    def chat(self, request: object) -> None:
        raise self.error

    def explain(self, *args: object) -> None:
        raise self.error


def stub_client(error: Exception) -> TestClient:
    stub = StubService(error)
    app = api.create_app(demo.DemoEngine(), stub, secrets=[SECRET])  # type: ignore[arg-type]
    register_chat_routes(app, stub, secrets=[SECRET])  # type: ignore[arg-type]
    return TestClient(app, raise_server_exceptions=False)


def test_chat_ok() -> None:
    client, _ = make_client([good_answer()])
    response = client.post("/api/chat", json=chat_json())
    assert response.status_code == 200
    body = ChatResponse.model_validate(response.json())
    assert body.reply == "Both documents list JavaScript and CSS." and body.explainer.kind == "agent"
    assert client.get("/api/health").json()["explainer"]["kind"] == "agent"


def test_chat_errors() -> None:
    client, _ = make_client([])
    cases = [
        (chat_json() | {"message": ""}, 422, "invalid_request", "message"),
        (chat_json() | {"message": "x" * 1001}, 422, "invalid_request", "message"),
        (chat_json() | {"history": [{"role": "user", "content": "a"}] * 21}, 422, "invalid_request", "history"),
        (chat_json(focus_id="j-404"), 404, "not_found", None),
        (chat_json(focus_id="r-2"), 400, "side_mismatch", "focus_id"),
    ]
    for body, status, code, field in cases:
        response = client.post("/api/chat", json=body)
        error = response.json()["error"]
        assert response.status_code == status and error["code"] == code
        if field:
            assert error["field"] == field


def test_chat_not_registered() -> None:
    agent, _, _ = make_agent([])
    app = api.create_app(demo.DemoEngine(), agent)
    client = TestClient(app)
    response = client.post("/api/chat", json=chat_json())
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"
    registered, _ = make_client([])
    assert registered.post("/api/nope", json={}).json()["error"]["code"] == "not_found"
    assert registered.get("/api/nope").status_code == 404


@pytest.mark.parametrize("reason", REASONS)
def test_failures_are_502_envelopes(reason: str) -> None:
    client = stub_client(AgentFailure(reason))  # type: ignore[arg-type]
    message, hint = FAILURE_COPY[reason]  # type: ignore[index]
    expected = {"error": {"code": "assistant_failed", "message": message, "hint": hint, "field": None, "reason": reason}}
    explain_body = explain_json()
    for response in (client.post("/api/chat", json=chat_json()), client.post("/api/explain", json=explain_body)):
        assert response.status_code == 502 and response.json() == expected
        assert SECRET not in response.text


def test_scripted_failures_are_502_envelopes() -> None:
    client, _ = make_client(["not json"] * 6)
    message, hint = FAILURE_COPY["invalid_reply"]
    for path, body in (("/api/chat", chat_json()), ("/api/explain", explain_json())):
        response = client.post(path, json=body)
        assert response.status_code == 502
        assert response.json()["error"] == {
            "code": "assistant_failed", "message": message, "hint": hint, "field": None,
            "reason": "invalid_reply",
        }  # fmt: skip
        assert "not json" not in response.text


def test_explain_route_with_agent() -> None:
    client, _ = make_client([good_answer()])
    response = client.post("/api/explain", json=explain_json())
    assert response.status_code == 200
    body = response.json()
    assert body["explainer"]["kind"] == "agent" and body["explainer"]["backend"] == "voyager"
    assert body["summary"] == "Both documents list JavaScript and CSS."
    assert body["claims"] and body["facts"]


def test_chat_privacy_and_internal_error(caplog: pytest.LogCaptureFixture) -> None:
    client, _ = make_client(["nope"] * 3)
    with caplog.at_level(logging.DEBUG):
        failed = client.post("/api/chat", json=chat_json(message=SECRET_MESSAGE))
    assert failed.status_code == 502
    messages = [record.getMessage() for record in caplog.records]
    assert sum(m.startswith("agent failed: reason=invalid_reply detail=") for m in messages) == 1
    for text in messages:
        assert SECRET_MESSAGE not in text and "Front-end developer" not in text
    boom = stub_client(RuntimeError(f"boom {SECRET_MESSAGE}"))
    response = boom.post("/api/chat", json=chat_json())
    assert response.status_code == 500 and response.json()["error"]["code"] == "internal"
    assert "boom" not in response.text and SECRET_MESSAGE not in response.text

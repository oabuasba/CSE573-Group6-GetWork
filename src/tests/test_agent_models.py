import json

import pytest
from pydantic import ValidationError

from getwork.agent import models

pytestmark = pytest.mark.usefixtures("offline")

QUERY = {
    "id": None,
    "side": "resume",
    "title": "Dev",
    "level": "Mid",
    "level_source": "listed",
    "years_experience": 3.0,
    "skills": [{"skill": "React", "cluster": "Frontend", "via": "taxonomy"}],
}


def chat_request(**overrides: object) -> dict:
    base = {
        "direction": "resume_to_posting",
        "query": QUERY,
        "focus_id": "j-1",
        "matches": [{"id": "j-1", "rank": 1, "score": 0.5}],
        "message": "Why is j-1 first?",
    }
    return {**base, **overrides}


def make_claim() -> models.Claim:
    return models.Claim(text="Both list React.", doc_ids=["query", "j-1"], skills=["React"], facts=["f1"])


def make_usage() -> models.Usage:
    return models.Usage(llm_calls=1, tool_calls=0, verify_retries=0, elapsed_ms=5)


def test_chat_request_limits() -> None:
    parsed = models.ChatRequest.model_validate(chat_request(message="  hi  "))
    assert parsed.message == "hi"
    models.ChatRequest.model_validate(chat_request(message="x" * 1000))
    models.ChatRequest.model_validate(chat_request(history=[{"role": "user", "content": "a"}] * 20))
    models.ChatRequest.model_validate(
        chat_request(matches=[{"id": "j", "rank": 1, "score": 0.1}] * 50)
    )
    bad = [
        {"message": ""},
        {"message": "   \n"},
        {"message": "x" * 1001},
        {"history": [{"role": "user", "content": "a"}] * 21},
        {"history": [{"role": "user", "content": "a" * 2001}]},
        {"history": [{"role": "system", "content": "a"}]},
        {"matches": [{"id": "j", "rank": 1, "score": 0.1}] * 51},
        {"unknown": 1},
        {"matches": [{"id": "j", "rank": 0, "score": 0.1}]},
        {"matches": [{"id": "j", "rank": -2, "score": 0.1}]},
    ]
    for overrides in bad:
        with pytest.raises(ValidationError):
            models.ChatRequest.model_validate(chat_request(**overrides))


def test_claim_limits() -> None:
    models.Claim(text="x" * 400, doc_ids=[], skills=[], facts=[])
    for text in ("", "x" * 401):
        with pytest.raises(ValidationError):
            models.Claim(text=text, doc_ids=[], skills=[], facts=[])


def test_models_round_trip() -> None:
    fact = models.Fact(id="f1", kind="shared_skills", doc_ids=["query"], skills=["React"], text="t")
    trace = models.ToolTrace(tool="compare_matches", args={"a": "j-1"}, ok=True, fact_ids=["f1"])
    response = models.ChatResponse(
        reply="Both list React.",
        claims=[make_claim()],
        facts=[fact],
        suggested_questions=["Why?"],
        trace=[trace],
        usage=make_usage(),
        explainer={"kind": "agent", "model": "m", "backend": "voyager", "prompt_version": "agent_v1"},
    )
    for model in (fact, trace, make_claim(), make_usage(), response):
        dumped = model.model_dump()
        assert type(model).model_validate(dumped).model_dump() == dumped
    request = models.ChatRequest.model_validate(chat_request())
    assert models.ChatRequest.model_validate(request.model_dump()) == request


def test_as_record_keys_and_json() -> None:
    result = models.AgentResult(
        status="failed",
        claims=[make_claim()],
        facts=[],
        trace=[models.ToolTrace(tool="t", args={}, ok=False, error="bad")],
        usage=make_usage(),
        model="m",
        backend="voyager",
        prompt_version="agent_v1",
        failure_reason="timeout",
    )
    record = result.as_record()
    assert set(record) == {
        "status", "failure_reason", "off_topic", "claims", "facts", "trace",
        "usage", "model", "backend", "prompt_version",
    }  # fmt: skip
    assert json.loads(json.dumps(record)) == record
    assert set(record["claims"][0]) == {"text", "doc_ids", "skills", "facts"}
    assert record["claims"][0]["text"] == "Both list React."


def test_chat_response_has_no_fallback_fields() -> None:
    assert "source" not in models.ChatResponse.model_fields
    assert "fallback_reason" not in models.ChatResponse.model_fields

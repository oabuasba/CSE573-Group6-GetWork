import json
from typing import get_args

import pytest
from pydantic import ValidationError

from getwork.agent import models, prompt

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


def make_usage() -> models.Usage:
    return models.Usage(llm_calls=1, tool_calls=0, elapsed_ms=5)


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


def test_failure_reasons_have_no_unverified() -> None:
    assert set(get_args(models.FailureReason)) == {
        "timeout", "budget_exhausted", "llm_unavailable", "model_rejected", "invalid_reply",
    }  # fmt: skip
    for gone in ("Claim", "OMITTED_NOTE", "MAX_CLAIMS", "MAX_CLAIM_CHARS", "MAX_VERIFY_RETRIES"):
        assert not hasattr(models, gone)
    assert models.MAX_REPLY_CHARS == 4000


def test_default_prompt_is_agent_v3() -> None:
    assert models.PROMPT_VERSION == "agent_v3"
    assert prompt.load_system_prompt()


def test_usage_and_chat_response_keys() -> None:
    assert set(models.Usage.model_fields) == {"llm_calls", "tool_calls", "elapsed_ms"}
    assert set(models.ChatResponse.model_fields) == {"reply", "trace", "usage", "explainer"}


def test_models_round_trip() -> None:
    fact = models.Fact(id="f1", kind="shared_skills", doc_ids=["query"], skills=["React"], text="t")
    trace = models.ToolTrace(tool="compare_matches", args={"a": "j-1"}, ok=True, fact_ids=["f1"])
    response = models.ChatResponse(
        reply="Both list React.",
        trace=[trace],
        usage=make_usage(),
        explainer={"kind": "agent", "model": "m", "backend": "voyager", "prompt_version": "agent_v3"},
    )
    for model in (fact, trace, make_usage(), response):
        dumped = model.model_dump()
        assert type(model).model_validate(dumped).model_dump() == dumped
    request = models.ChatRequest.model_validate(chat_request())
    assert models.ChatRequest.model_validate(request.model_dump()) == request


def make_result(**overrides: object) -> models.AgentResult:
    base = {
        "status": "failed",
        "reply": "",
        "facts": [],
        "trace": [models.ToolTrace(tool="t", args={}, ok=False, error="bad")],
        "usage": make_usage(),
        "model": "m",
        "backend": "voyager",
        "prompt_version": "agent_v3",
        "failure_reason": "timeout",
    }
    return models.AgentResult(**{**base, **overrides})


def test_agent_result_record_keys() -> None:
    result = make_result()
    record = result.as_record()
    assert set(record) == {
        "status", "failure_reason", "reply", "facts", "trace",
        "usage", "model", "backend", "prompt_version", "ungrounded_skills",
    }  # fmt: skip
    assert json.loads(json.dumps(record)) == record
    assert record["ungrounded_skills"] == []
    assert set(record["usage"]) == {"llm_calls", "tool_calls", "elapsed_ms"}
    assert make_result(ungrounded_skills=["Terraform"]).as_record()["ungrounded_skills"] == ["Terraform"]


def test_chat_response_has_no_fallback_fields() -> None:
    assert "source" not in models.ChatResponse.model_fields
    assert "fallback_reason" not in models.ChatResponse.model_fields

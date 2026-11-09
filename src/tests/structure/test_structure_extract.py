from collections.abc import Callable
from typing import Any

import pytest

from getwork.llm.fake import FakeLLMClient, make_output_json
from getwork.llm.retry import Throttle
from getwork.structure.extract import ExtractConfig, Outcome, SourceDoc, structure_one
from getwork.structure.prompt import load_system_prompt

pytestmark = pytest.mark.usefixtures("offline")

DOC = SourceDoc(id="r1", source="resumes", title=None, text="Python developer")


def _throttle() -> Throttle:
    return Throttle(clock=lambda: 0.0, sleep=lambda s: None)


def _run(client: FakeLLMClient, cfg: ExtractConfig, doc: SourceDoc = DOC) -> Outcome:
    return structure_one(
        client=client, cfg=cfg, throttle=_throttle(), doc=doc, sleep=lambda s: None
    )


def test_validation_retry_sends_error_feedback() -> None:
    client = FakeLLMClient(script=["not json", make_output_json()])
    outcome = _run(client, ExtractConfig(model="m"))
    assert outcome.failure is None
    assert outcome.record["id"] == "r1"
    assert len(client.calls) == 2
    first, second = client.calls[0].user, client.calls[1].user
    assert second.startswith(first)
    assert "rejected" in second and len(second) > len(first)
    assert outcome.stats.calls == 2


def test_validation_exhausted_returns_failure() -> None:
    client = FakeLLMClient(script=["bad", "bad", "bad"])
    outcome = _run(client, ExtractConfig(model="m"))
    assert outcome.record is None
    assert outcome.failure.stage == "validation"
    assert outcome.failure.error_type == "OutputValidationError"
    assert outcome.failure.attempts == 3
    assert outcome.failure.message
    assert len(client.calls) == 3


@pytest.mark.parametrize("use_schema", [True, False])
@pytest.mark.parametrize("temperature,effort", [(0.0, "none"), (None, None), (0.7, "high")])
def test_request_fields_follow_config(
    use_schema: bool, temperature: float | None, effort: str | None
) -> None:
    cfg = ExtractConfig(
        model="m-x",
        prompt_version="extract_v1",
        temperature=temperature,
        max_tokens=123,
        reasoning_effort=effort,
        use_response_schema=use_schema,
    )
    client = FakeLLMClient(script=[make_output_json()])
    outcome = _run(client, cfg)
    request = client.calls[0]
    assert request.system == load_system_prompt()
    assert request.model == "m-x"
    assert request.temperature == temperature
    assert request.max_tokens == 123
    assert request.reasoning_effort == effort
    assert (request.response_schema is not None) is use_schema
    assert outcome.record["prompt_version"] == cfg.prompt_version
    assert outcome.record["model"] == "m-x"


def test_backend_in_record_and_extra_body_in_requests() -> None:
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    cfg = ExtractConfig(model="m", backend="local", extra_body=extra)
    client = FakeLLMClient(script=["not json", make_output_json()])
    outcome = _run(client, cfg)
    assert outcome.record["backend"] == "local"
    assert len(client.calls) == 2
    assert all(call.extra_body == extra for call in client.calls)
    assert _run(FakeLLMClient(script=[make_output_json()]), ExtractConfig(model="m")).record[
        "backend"
    ] == "voyager"

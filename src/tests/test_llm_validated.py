import pytest

from getwork.llm.client import (
    FatalLLMError,
    LLMRequest,
    PermanentLLMError,
    TransientLLMError,
)
from getwork.llm.fake import FakeLLMClient
from getwork.llm.retry import CallStats, RetryConfig, Throttle
from getwork.llm.validated import (
    ValidationExhausted,
    complete_validated,
    parse_json_object,
    strip_reply_wrappers,
)

pytestmark = pytest.mark.usefixtures("offline")

REQUEST = LLMRequest(
    system="s",
    user="original question",
    model="m",
    temperature=None,
    max_tokens=10,
    reasoning_effort=None,
    response_schema=None,
)


def _run(client: FakeLLMClient, parse, cfg: RetryConfig | None = None) -> tuple[object, CallStats]:
    stats = CallStats()
    result = complete_validated(
        client,
        REQUEST,
        parse,
        cfg=cfg or RetryConfig(),
        throttle=Throttle(clock=lambda: 0.0, sleep=lambda s: None),
        sleep=lambda s: None,
        stats=stats,
    )
    return result, stats


def _must_be_ok(text: str) -> str:
    if text != "ok":
        raise ValueError("x" * 400)
    return text


def test_strip_reply_wrappers() -> None:
    assert strip_reply_wrappers("  hi  ") == "hi"
    assert strip_reply_wrappers("<think>a\nb\nc</think>\n{\"a\": 1}") == '{"a": 1}'
    assert strip_reply_wrappers('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_reply_wrappers('```\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_reply_wrappers('<think>t\nu</think>\n```json\n{"a": 1}\n```\n') == '{"a": 1}'


def test_parse_json_object_errors() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError, match="^empty reply"):
        parse_json_object("<think>only</think>  ")
    with pytest.raises(ValueError, match="^not valid JSON"):
        parse_json_object("{nope")
    with pytest.raises(ValueError, match="^expected a JSON object"):
        parse_json_object("[1, 2]")


def test_first_reply_valid() -> None:
    client = FakeLLMClient(script=["ok"])
    result, stats = _run(client, _must_be_ok)
    assert result == "ok"
    assert len(client.calls) == 1
    assert client.calls[0] == REQUEST
    assert stats.calls == 1


def test_feedback_retry() -> None:
    client = FakeLLMClient(script=["bad", "ok"])
    result, _ = _run(client, _must_be_ok)
    assert result == "ok"
    assert len(client.calls) == 2
    second = client.calls[1].user
    assert second.startswith(REQUEST.user)
    assert "x" * 300 in second
    assert "x" * 301 not in second


def test_exhausted_raises() -> None:
    client = FakeLLMClient(script=["bad", "bad", "bad"])
    with pytest.raises(ValidationExhausted) as info:
        _run(client, _must_be_ok)
    assert info.value.attempts == 3
    assert info.value.last_error == "x" * 300
    assert len(client.calls) == 3
    assert client.calls[2].user.count("Your previous reply was rejected") == 1


def test_transient_error_retried() -> None:
    client = FakeLLMClient(script=[TransientLLMError("x"), "ok"])
    result, stats = _run(client, _must_be_ok, cfg=RetryConfig(max_attempts=2))
    assert result == "ok"
    assert stats.retries == 1
    assert stats.calls == 2


@pytest.mark.parametrize("error_type", [PermanentLLMError, FatalLLMError])
def test_permanent_and_fatal_propagate(error_type: type[Exception]) -> None:
    client = FakeLLMClient(script=[error_type("nope"), "ok"])
    with pytest.raises(error_type):
        _run(client, _must_be_ok)
    assert len(client.calls) == 1

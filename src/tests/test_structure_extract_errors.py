import pytest

from getwork.llm.client import (
    MAX_ERROR_CHARS,
    FatalLLMError,
    LLMRequest,
    PermanentLLMError,
    RateLimitError,
    TransientLLMError,
)
from getwork.llm.fake import FakeLLMClient, make_output_json
from getwork.llm.retry import RetryConfig, Throttle
from getwork.structure.extract import ExtractConfig, Outcome, SourceDoc, structure_one

pytestmark = pytest.mark.usefixtures("offline")

DOC = SourceDoc(id="r1", source="resumes", title=None, text="Python developer")
SECRET = "sk-super-secret-key"


def _run(
    client: FakeLLMClient,
    cfg: ExtractConfig,
    doc: SourceDoc = DOC,
    secrets: tuple[str, ...] = (),
    throttle: Throttle | None = None,
) -> Outcome:
    return structure_one(
        client=client,
        cfg=cfg,
        throttle=throttle or Throttle(clock=lambda: 0.0, sleep=lambda s: None),
        doc=doc,
        secrets=secrets,
        sleep=lambda s: None,
    )


def test_long_input_truncated_and_flagged() -> None:
    cfg = ExtractConfig(model="m", max_input_chars=100)
    long_doc = SourceDoc(id="r1", source="resumes", title=None, text="x" * 500)
    client = FakeLLMClient(script=[make_output_json(), make_output_json()])
    long_outcome = _run(client, cfg, long_doc)
    short_outcome = _run(client, cfg)
    assert long_outcome.truncated is True
    assert short_outcome.truncated is False
    assert client.calls[0].user.count("x") == 100
    assert "Python developer" in client.calls[1].user


def test_validation_third_attempt_built_from_original_message() -> None:
    client = FakeLLMClient(script=["bad one", "bad two", make_output_json()])
    outcome = _run(client, ExtractConfig(model="m"))
    assert outcome.failure is None
    first, second, third = (c.user for c in client.calls)
    assert second.startswith(first) and third.startswith(first)
    assert len(third) <= len(second) + len("bad two") + 5
    assert second.count("rejected") == 1 and third.count("rejected") == 1


@pytest.mark.parametrize(
    "stage,error,cfg",
    [
        (
            "transient",
            TransientLLMError("down", status=503),
            ExtractConfig(model="m", retry=RetryConfig(max_attempts=2)),
        ),
        (
            "rate_limit",
            RateLimitError("slow", status=429),
            ExtractConfig(model="m", retry=RetryConfig(max_rate_limit_waits=1)),
        ),
        ("permanent", PermanentLLMError("bad request", status=400), ExtractConfig(model="m")),
    ],
)
def test_error_stages_become_failures(
    stage: str, error: Exception, cfg: ExtractConfig
) -> None:
    now = [0.0]

    def advance(seconds: float) -> None:
        now[0] += seconds

    throttle = Throttle(clock=lambda: now[0], sleep=advance)
    client = FakeLLMClient(handler=lambda request: (_ for _ in ()).throw(error))
    outcome = _run(client, cfg, throttle=throttle)
    assert outcome.record is None
    assert outcome.failure.stage == stage
    assert outcome.failure.error_type == type(error).__name__
    assert outcome.failure.attempts == len(client.calls)
    assert outcome.failure.message


def test_fatal_error_propagates() -> None:
    client = FakeLLMClient(script=[FatalLLMError("nope", status=401)])
    with pytest.raises(FatalLLMError):
        _run(client, ExtractConfig(model="m"))


def _raiser(error: Exception):
    def handler(request: LLMRequest) -> str:
        raise error

    return handler


@pytest.mark.parametrize(
    "error,cfg",
    [
        (TransientLLMError(f"oops {SECRET} " + "z" * 900), ExtractConfig(model="m", retry=RetryConfig(max_attempts=1))),
        (PermanentLLMError(f"oops {SECRET} " + "z" * 900), ExtractConfig(model="m")),
        (KeyError(f"{SECRET} " + "z" * 900), ExtractConfig(model="m")),
    ],
)
def test_failure_messages_are_redacted(error: Exception, cfg: ExtractConfig) -> None:
    outcome = _run(FakeLLMClient(handler=_raiser(error)), cfg, secrets=(SECRET,))
    message = outcome.failure.message
    assert SECRET not in message
    assert "***" in message
    assert len(message) <= MAX_ERROR_CHARS


def test_validation_failure_message_is_redacted_and_cut() -> None:
    client = FakeLLMClient(script=[f"{SECRET} " + "q" * 900] * 3)
    outcome = _run(client, ExtractConfig(model="m"), secrets=(SECRET,))
    assert outcome.failure.stage == "validation"
    assert SECRET not in outcome.failure.message
    assert len(outcome.failure.message) <= MAX_ERROR_CHARS


def test_unexpected_exception_becomes_permanent_failure() -> None:
    client = FakeLLMClient(handler=_raiser(KeyError("boom")))
    outcome = _run(client, ExtractConfig(model="m"))
    assert outcome.failure.stage == "permanent"
    assert outcome.failure.error_type == "KeyError"
    assert outcome.failure.attempts == 1
    assert len(client.calls) == 1

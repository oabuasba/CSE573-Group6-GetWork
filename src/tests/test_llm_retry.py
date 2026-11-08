import pytest

from getwork.llm.client import (
    FatalLLMError,
    LLMRequest,
    PermanentLLMError,
    RateLimitError,
    TransientLLMError,
)
from getwork.llm.retry import (
    CallStats,
    RetriesExhausted,
    RetryConfig,
    Throttle,
    call_with_retry,
)

pytestmark = pytest.mark.usefixtures("offline")

REQUEST = LLMRequest(
    system="s",
    user="u",
    model="m",
    temperature=None,
    max_tokens=10,
    reasoning_effort=None,
    response_schema=None,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class ScriptedClient:
    """Returns strings and raises exceptions from a script, in order."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.calls.append(request)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def run(script: list, cfg: RetryConfig = RetryConfig()):
    clock = FakeClock()
    client = ScriptedClient(script)
    stats = CallStats()
    throttle = Throttle(clock=clock.time, sleep=clock.sleep)
    try:
        result = call_with_retry(client, REQUEST, cfg, throttle, clock.sleep, stats)
    except Exception as e:
        result = e
    return result, clock, client, stats


def test_transient_backoff_schedule() -> None:
    result, clock, client, stats = run([TransientLLMError("x")] * 5)
    assert isinstance(result, RetriesExhausted)
    assert result.kind == "transient"
    assert result.attempts == 5
    assert clock.sleeps == [5.0, 10.0, 20.0, 40.0]
    assert (stats.calls, stats.retries) == (5, 4)


def test_retry_after_longer_than_backoff_wins() -> None:
    _, clock, _, _ = run([TransientLLMError("x", retry_after=30), "ok"])
    assert clock.sleeps == [30.0]
    _, clock, _, _ = run([TransientLLMError("x", retry_after=1), "ok"])
    assert clock.sleeps == [5.0]


def test_success_after_transient_returns_text() -> None:
    result, _, client, stats = run([TransientLLMError("x"), "hello"])
    assert result == "hello"
    assert (stats.calls, stats.retries) == (2, 1)


def test_permanent_error_not_retried() -> None:
    result, _, client, stats = run([PermanentLLMError("bad"), "ok"])
    assert isinstance(result, PermanentLLMError)
    assert len(client.calls) == 1
    assert stats.calls == 1


def test_fatal_error_propagates() -> None:
    result, _, client, stats = run([FatalLLMError("denied", status=401), "ok"])
    assert isinstance(result, FatalLLMError)
    assert len(client.calls) == 1
    assert stats.calls == 1


def test_throttle_shares_pause_between_callers() -> None:
    clock = FakeClock()
    throttle = Throttle(clock=clock.time, sleep=clock.sleep)
    throttle.block_for(60)
    clock.now = 30
    throttle.wait()
    assert clock.sleeps == [30.0]
    assert clock.now == 60
    throttle.block_for(10)
    assert throttle._resume_at == 70
    throttle.block_for(5)
    assert throttle._resume_at == 70


def test_throttle_wait_honors_extended_pause() -> None:
    clock = FakeClock()
    calls: list[float] = []

    def sleep(seconds: float) -> None:
        calls.append(seconds)
        clock.now += seconds
        if len(calls) == 1:
            throttle.block_for(10)  # another worker extends the pause mid-sleep

    throttle = Throttle(clock=clock.time, sleep=sleep)
    throttle.block_for(60)
    throttle.wait()
    assert len(calls) == 2
    assert calls[0] == 60.0
    assert calls[1] == pytest.approx(10.0)


def test_rate_limit_waits_then_succeeds() -> None:
    result, clock, _, stats = run([RateLimitError("slow", status=429), "ok"])
    assert result == "ok"
    assert sum(clock.sleeps) >= 60.0
    assert (stats.calls, stats.retries, stats.rate_limited) == (2, 1, 1)


def test_rate_limit_exhausted_raises() -> None:
    cfg = RetryConfig(max_rate_limit_waits=2)
    result, _, _, stats = run([RateLimitError("slow", status=429)] * 3, cfg)
    assert isinstance(result, RetriesExhausted)
    assert result.kind == "rate_limit"
    assert result.attempts == 3
    assert stats.calls == 3
    assert stats.rate_limited == 3


def test_retry_config_defaults() -> None:
    cfg = RetryConfig()
    assert cfg.max_attempts == 5
    assert cfg.max_rate_limit_waits == 5
    assert cfg.rate_limit_wait_s == 60.0

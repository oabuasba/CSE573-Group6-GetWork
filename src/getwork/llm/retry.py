"""Retry, backoff, and a pause shared by all workers for rate-limit responses."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from getwork.llm.client import (
    LLMClient,
    LLMError,
    LLMRequest,
    RateLimitError,
    TransientLLMError,
)


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 5  # model calls per request for TransientLLMError (first call included)
    backoff_base_s: float = 5.0
    backoff_cap_s: float = 80.0
    rate_limit_wait_s: float = 60.0
    max_rate_limit_waits: int = 5  # 429 responses tolerated per request before giving up


@dataclass
class CallStats:
    calls: int = 0  # client.complete invocations
    retries: int = 0  # invocations that were repeats after a transient or 429 error
    rate_limited: int = 0  # 429 responses seen


class RetriesExhausted(Exception):
    def __init__(self, kind: str, attempts: int, last_error: LLMError) -> None:
        super().__init__(f"{kind} retries exhausted after {attempts} calls: {last_error}")
        self.kind = kind  # "transient" | "rate_limit"
        self.attempts = attempts
        self.last_error = last_error


class Throttle:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._resume_at = float("-inf")

    def block_for(self, seconds: float) -> None:
        """resume_at = max(resume_at, clock() + seconds). Thread-safe."""
        with self._lock:
            self._resume_at = max(self._resume_at, self._clock() + seconds)

    def wait(self) -> None:
        """Sleep until the shared pause ends, re-checking so an extended pause is honored."""
        while True:
            with self._lock:
                remaining = self._resume_at - self._clock()
            if remaining <= 0:
                return
            self._sleep(remaining)


def backoff_delay(attempt: int, cfg: RetryConfig, retry_after: float | None) -> float:
    """attempt is the 1-based number of the call that just failed."""
    exponential = min(cfg.backoff_cap_s, cfg.backoff_base_s * 2 ** (attempt - 1))
    return max(exponential, retry_after or 0.0)


def call_with_retry(
    client: LLMClient,
    request: LLMRequest,
    cfg: RetryConfig,
    throttle: Throttle,
    sleep: Callable[[float], None],
    stats: CallStats,
) -> str:
    """Call the model, waiting out transient and 429 errors.

    Permanent and fatal errors propagate unchanged. 429 waits go through the shared
    throttle (performed by the next loop's throttle.wait()); transient backoff sleeps locally.
    """
    transient_failures = 0
    rate_limit_failures = 0
    request_calls = 0
    while True:
        throttle.wait()
        stats.calls += 1
        request_calls += 1
        try:
            return client.complete(request)
        except TransientLLMError as e:
            transient_failures += 1
            if transient_failures >= cfg.max_attempts:
                raise RetriesExhausted("transient", request_calls, e) from e
            stats.retries += 1
            sleep(backoff_delay(transient_failures, cfg, e.retry_after))
        except RateLimitError as e:
            stats.rate_limited += 1
            rate_limit_failures += 1
            if rate_limit_failures > cfg.max_rate_limit_waits:
                raise RetriesExhausted("rate_limit", request_calls, e) from e
            stats.retries += 1
            throttle.block_for(max(cfg.rate_limit_wait_s, e.retry_after or 0.0))

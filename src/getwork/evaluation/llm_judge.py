"""A judge backed by an LLMClient: schema-first request, plain-JSON fallback, error mapping."""

from __future__ import annotations

import dataclasses
import threading
import time
from collections.abc import Callable, Mapping

from getwork.evaluation import judges
from getwork.evaluation.judges import (
    JudgeError,
    JudgeFatalError,
    JudgeInfo,
    PairText,
    Rubric,
    Verdict,
)
from getwork.llm.client import FatalLLMError, LLMClient, LLMError, LLMRequest, PermanentLLMError
from getwork.llm.retry import CallStats, RetriesExhausted, RetryConfig, Throttle
from getwork.llm.validated import ValidationExhausted, complete_validated


class LLMJudge:
    def __init__(
        self,
        client: LLMClient,
        info: JudgeInfo,
        rubric: Rubric,
        *,
        max_chars: int = 20000,
        max_tokens: int = 512,
        reasoning_effort: str | None = "none",
        use_schema: bool = True,
        extra_body: Mapping[str, object] | None = None,
        retry: RetryConfig = RetryConfig(),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._info = info
        self._rubric = rubric
        self._max_chars = max_chars
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._use_schema = use_schema
        self._extra_body = extra_body
        self._retry = retry
        self._sleep = sleep
        self._lock = threading.Lock()
        self._fallback = False

    @property
    def info(self) -> JudgeInfo:
        return self._info

    @property
    def schema_fallback(self) -> bool:
        with self._lock:
            return self._fallback

    def judge(self, pair: PairText) -> tuple[Verdict, bool]:
        system, user, truncated = judges.build_prompt(self._rubric, pair, self._max_chars)
        request = LLMRequest(
            system=system,
            user=user,
            model=self._info.model,
            temperature=0,
            max_tokens=self._max_tokens,
            reasoning_effort=self._reasoning_effort,
            response_schema=judges.VERDICT_SCHEMA if self._sends_schema() else None,
            extra_body=self._extra_body,
        )
        try:
            return self._call(request), truncated
        except ValidationExhausted as e:
            raise JudgeError("invalid_reply", e.last_error) from e
        except PermanentLLMError as e:
            raise JudgeError("rejected", f"status {e.status}") from e
        except FatalLLMError as e:
            raise JudgeFatalError(f"status {e.status}") from e
        except RetriesExhausted as e:
            raise JudgeError("unavailable", e.kind) from e
        except LLMError as e:
            raise JudgeError("unavailable", f"status {e.status}") from e

    def _sends_schema(self) -> bool:
        return self._use_schema and not self.schema_fallback

    def _call(self, request: LLMRequest) -> Verdict:
        try:
            return self._ask(request)
        except PermanentLLMError:
            if request.response_schema is None:
                raise
        verdict = self._ask(dataclasses.replace(request, response_schema=None))
        with self._lock:
            self._fallback = True
        return verdict

    def _ask(self, request: LLMRequest) -> Verdict:
        return complete_validated(
            self._client,
            request,
            judges.parse_verdict,
            cfg=self._retry,
            throttle=Throttle(),
            sleep=self._sleep,
            stats=CallStats(),
        )

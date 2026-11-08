"""Validate-and-regenerate loop around the retrying model call (no openai import)."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable
from typing import TypeVar

from getwork.llm.client import LLMClient, LLMRequest
from getwork.llm.retry import CallStats, RetryConfig, Throttle, call_with_retry

T = TypeVar("T")

MAX_FEEDBACK_ERROR_CHARS = 300
_THINK_BLOCK = re.compile(r"\A<think>.*?</think>", re.DOTALL)
_CODE_FENCE = re.compile(r"\A```(?:json)?[ \t]*\n?(.*?)\n?```\Z", re.DOTALL)


def strip_reply_wrappers(text: str) -> str:
    """Trim; drop one leading think block and one surrounding code fence; trim again."""
    text = _THINK_BLOCK.sub("", text.strip(), count=1).strip()
    fenced = _CODE_FENCE.match(text)
    return fenced.group(1).strip() if fenced else text


def parse_json_object(text: str) -> dict:
    cleaned = strip_reply_wrappers(text)
    if not cleaned:
        raise ValueError("empty reply")
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError("not valid JSON") from e
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


class ValidationExhausted(Exception):
    def __init__(self, last_error: str, attempts: int) -> None:
        super().__init__(f"reply rejected after {attempts} attempts: {last_error}")
        self.last_error = last_error[:MAX_FEEDBACK_ERROR_CHARS]
        self.attempts = attempts


def build_feedback(user: str, error: str) -> str:
    return (
        user
        + "\n\nYour previous reply was rejected: "
        + error[:MAX_FEEDBACK_ERROR_CHARS]
        + "\nReturn only the corrected JSON object."
    )


def complete_validated(
    client: LLMClient,
    request: LLMRequest,
    parse: Callable[[str], T],
    *,
    cfg: RetryConfig,
    throttle: Throttle,
    sleep: Callable[[float], None],
    stats: CallStats,
    max_attempts: int = 3,
) -> T:
    """Call and parse; on ValueError retry with the error appended to the original user text."""
    last_error = ""
    current = request
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            current = dataclasses.replace(request, user=build_feedback(request.user, last_error))
        text = call_with_retry(client, current, cfg, throttle, sleep, stats)
        try:
            return parse(text)
        except ValueError as e:
            last_error = str(e)
    raise ValidationExhausted(last_error, max_attempts)

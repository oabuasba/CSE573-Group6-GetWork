"""LLM client seam: request shape, error taxonomy, and secret redaction (no third-party imports)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

MAX_ERROR_CHARS: int = 500
MAX_RETRY_AFTER_S: float = 300.0
REDACTION = "***"

TRANSIENT_STATUSES = frozenset({408})
FATAL_STATUSES = frozenset({401, 403, 404, 410})


@dataclass(frozen=True)
class LLMRequest:
    system: str
    user: str
    model: str
    temperature: float | None  # None = do not send
    max_tokens: int
    reasoning_effort: str | None  # "none" | "low" | "medium" | "high"; None = do not send
    response_schema: dict | None  # JSON Schema for strict structured output; None = plain text
    extra_body: Mapping[str, object] | None = None  # merged into the SDK extra_body; None = nothing


class LLMError(Exception):
    def __init__(
        self, message: str, status: int | None = None, retry_after: float | None = None
    ) -> None:
        """Set self.status and self.retry_after. str(self) is message."""
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class TransientLLMError(LLMError):
    """Retry with backoff: connection error, timeout, 408, 5xx."""


class RateLimitError(LLMError):
    """HTTP 429."""


class PermanentLLMError(LLMError):
    """This record cannot succeed unchanged: 400, 413, 422, other statuses."""


class FatalLLMError(LLMError):
    """The whole run cannot succeed: 401, 403, 404, 410."""


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> str:
        """Return the assistant message content ("" if there is none). Raise only LLMError subclasses."""
        ...


def redact_secrets(text: str, secrets: Sequence[str]) -> str:
    """Replace every occurrence of every non-empty secret in text with "***", longest secret first."""
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, REDACTION)
    return text


def _redact_and_truncate(text: str, secrets: Sequence[str]) -> str:
    # Redact before truncating so a secret cut at the boundary cannot survive.
    return redact_secrets(text, secrets)[:MAX_ERROR_CHARS]


def describe_exception(exc: BaseException, secrets: Sequence[str] = ()) -> str:
    """f"{type(exc).__name__}: {exc}", redacted with secrets, then truncated to MAX_ERROR_CHARS."""
    return _redact_and_truncate(f"{type(exc).__name__}: {exc}", secrets)


def _parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    if not headers:
        return None
    lowered = {name.lower(): value for name, value in headers.items()}
    raw = lowered.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    if not seconds >= 0:  # also rejects NaN
        return None
    return min(seconds, MAX_RETRY_AFTER_S)


def _error_class(status: int | None) -> type[LLMError]:
    if status is None or status in TRANSIENT_STATUSES or 500 <= status <= 599:
        return TransientLLMError
    if status == 429:
        return RateLimitError
    if status in FATAL_STATUSES:
        return FatalLLMError
    return PermanentLLMError


def error_from_http(
    status: int | None,
    headers: Mapping[str, str] | None,
    message: str,
    secrets: Sequence[str] = (),
) -> LLMError:
    """Map an HTTP status (None = no response) to an LLMError subclass.

    The message is redacted then truncated; retry_after comes from a case-insensitive
    "retry-after" header when numeric and >= 0, capped at MAX_RETRY_AFTER_S.
    """
    return _error_class(status)(
        _redact_and_truncate(message, secrets),
        status=status,
        retry_after=_parse_retry_after(headers),
    )

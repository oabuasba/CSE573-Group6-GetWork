import re
from collections.abc import Sequence

from getwork.agent.models import FailureReason
from getwork.llm.client import redact_secrets

MAX_DETAIL_CHARS = 400
_BEARER = re.compile(r"(?i)\bbearer\s+\S+")
_URL = re.compile(r"https?://\S+")
_KEY_LIKE = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}")


def redact_detail(text: str, secrets: Sequence[str] = ()) -> str:
    """One short log-safe line: known secrets, bearer tokens, key-like tokens and whole URLs (host and query) removed."""
    cleaned = redact_secrets(text, secrets)
    for pattern, replacement in ((_BEARER, "Bearer ***"), (_URL, "<url>"), (_KEY_LIKE, "***")):
        cleaned = pattern.sub(replacement, cleaned)
    return " ".join(cleaned.split())[:MAX_DETAIL_CHARS]


FAILURE_COPY: dict[FailureReason, tuple[str, str]] = {
    "timeout": (
        "The assistant took too long to answer.",
        "Try again in a moment, or ask a shorter question.",
    ),
    "budget_exhausted": (
        "That question needed more steps than the assistant is allowed to take.",
        "Try asking about one thing at a time.",
    ),
    "llm_unavailable": (
        "The assistant could not be reached right now.",
        "Wait a moment and try again. If it keeps happening, ask whoever runs this server "
        "to check the assistant's service.",
    ),
    "model_rejected": (
        "The assistant's service did not accept the request.",
        "Ask whoever runs this server to check the assistant's settings, then try again.",
    ),
    "invalid_reply": (
        "The assistant gave a reply that could not be used.",
        "Try again, or ask the question in different words.",
    ),
    "unverified": (
        "The assistant could not give an answer that matched the skills in these two documents.",
        "Try asking in different words. The skills listed on the card are unaffected.",
    ),
}


def failure_copy(reason: FailureReason) -> tuple[str, str]:
    return FAILURE_COPY[reason]


class AgentFailure(Exception):
    reason: FailureReason
    message: str
    hint: str

    def __init__(self, reason: FailureReason, detail: str | None = None) -> None:
        message, hint = failure_copy(reason)
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.hint = hint
        self.detail = detail

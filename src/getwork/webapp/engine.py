"""The Engine protocol: where documents and vectors come from (CC-18, ADR-15)."""

from dataclasses import dataclass
from typing import Protocol

from getwork.webapp import models


class EngineError(Exception):
    """An expected failure with a stable code; status comes from models.ERROR_CODES."""

    def __init__(
        self, code: str, message: str, *, hint: str | None = None, field: str | None = None
    ) -> None:
        if code not in models.ERROR_CODES:
            raise ValueError(f"unknown engine error code: {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.field = field
        self.status = models.ERROR_CODES[code]


class EngineLoadError(RuntimeError):
    """Artifacts missing or inconsistent at startup; the message carries the fix hint."""


@dataclass(frozen=True)
class MatchResult:
    query: models.DocumentView
    matches: list[tuple[models.DocumentView, float]]  # rank order, rank = position + 1
    notes: list[models.Note]
    provenance: models.Provenance
    k_effective: int


class Engine(Protocol):
    def health(self) -> models.EngineHealth: ...

    def examples(self, side: models.Side, limit: int) -> list[models.DocumentView]: ...

    def get_document(self, doc_id: str) -> models.DocumentView:
        """EngineError("not_found") naming the id."""
        ...

    def match(self, request: models.MatchRequest) -> MatchResult:
        """Raises EngineError only (codes in ERROR_CODES); any other exception is a bug (500)."""
        ...

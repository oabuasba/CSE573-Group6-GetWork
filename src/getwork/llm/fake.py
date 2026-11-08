"""Offline LLMClient test double: scripted replies or a handler, recording every request."""

from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable, Sequence

from getwork.llm.client import LLMRequest


class FakeLLMClient:
    def __init__(
        self,
        handler: Callable[[LLMRequest], str] | None = None,
        script: Sequence[str | BaseException] | None = None,
        model_info: dict | None = None,
    ) -> None:
        """Exactly one of handler and script must be given, else ValueError."""
        if (handler is None) == (script is None):
            raise ValueError("give exactly one of handler and script")
        self._handler = handler
        self._script: deque[str | BaseException] | None = (
            deque(script) if script is not None else None
        )
        self._model_info = model_info
        self._lock = threading.Lock()
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        with self._lock:
            self.calls.append(request)
            item = self._pop_script() if self._script is not None else None
        if self._handler is not None:
            return self._handler(request)
        if isinstance(item, BaseException):
            raise item
        return item

    def _pop_script(self) -> str | BaseException:
        if not self._script:
            raise AssertionError("script exhausted")
        return self._script.popleft()

    def describe_model(self, model: str) -> dict:
        if self._model_info is not None:
            return self._model_info
        return {"id": model, "found": True}


def make_output_json(
    title: str = "Software Engineer",
    skills: Sequence[str] = ("Python",),
    experience: Sequence[dict] = (),
    education: Sequence[dict] = (),
    years_experience: float | None = None,
) -> str:
    return json.dumps(
        {
            "title": title,
            "skills": list(skills),
            "experience": list(experience),
            "education": list(education),
            "years_experience": years_experience,
        }
    )

"""The default judges and reasoning choices; the one place the judge model ids are written."""

from __future__ import annotations

from collections.abc import Mapping

DEFAULT_JUDGE_IDS: tuple[str, str] = ("judge-a", "judge-b")
DEFAULT_JUDGE_MODELS: tuple[str, str] = ("gemma4-31b-it", "llama4-maverick-17b")
DEFAULT_JUDGE_FAMILIES: tuple[str, str] = ("google", "meta")
# Structuring, relabel and chat-agent model: always checked by the independence rule.
DEFAULT_ALSO_USED_MODELS: tuple[str, ...] = ("qwen38-27b",)
JUDGE_CONTEXT_TOKENS: int = 131072
DEFAULT_REASONING: Mapping[str, str | None] = {"gemma4-31b-it": "none", "llama4-maverick-17b": None}
REASONING_CHOICES: tuple[str, ...] = ("auto", "none", "low", "medium", "high", "omit")


def resolve_reasoning(model: str, choice: str = "auto") -> str | None:
    """The reasoning_effort to send for model; None means the field is omitted."""
    if choice not in REASONING_CHOICES:
        raise ValueError(f"unknown reasoning choice {choice!r}; choose from {REASONING_CHOICES}")
    if choice == "auto":
        return DEFAULT_REASONING.get(model)
    return None if choice == "omit" else choice

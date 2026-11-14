"""System prompt loading, user-message building and action parsing for the agent."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from getwork.agent.models import MAX_REPLY_CHARS, PROMPT_VERSION, ChatMatchRef, ChatMessage, Fact
from getwork.agent.tools import TOOL_NAMES
from getwork.llm.validated import parse_json_object, strip_reply_wrappers
from getwork.webapp.models import DocumentView

MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_TOTAL_CHARS = 6000
NONE_TEXT = "(none)"


@dataclass(frozen=True)
class Action:
    kind: Literal["tool", "answer"]
    tool: str | None = None
    args: dict[str, object] | None = None
    text: str | None = None  # the answer text when kind == "answer"


@cache
def load_system_prompt(version: str = PROMPT_VERSION) -> str:
    """Reads prompts/{version}.md with functools.cache; FileNotFoundError if absent."""
    path = Path(__file__).parent / "prompts" / f"{version}.md"
    return path.read_text(encoding="utf-8")


def _documents_section(docs: Mapping[str, DocumentView]) -> str:
    data = {
        ref: {"title": doc.title, "level": doc.level, "skills": [[s.skill, s.cluster] for s in doc.skills]}
        for ref, doc in docs.items()
    }
    return json.dumps(data, ensure_ascii=False)


def _one_line(text: str) -> str:
    """Collapses newlines so data cannot start a line of its own in the prompt."""
    return " ".join(text.split())


def _history_section(history: Sequence[ChatMessage]) -> str:
    lines = [_one_line(f"{m.role}: {m.content}") for m in history[-MAX_HISTORY_MESSAGES:]]
    while len(lines) > 1 and len("\n".join(lines)) > MAX_HISTORY_TOTAL_CHARS:
        lines.pop(0)
    return "\n".join(lines) if lines else NONE_TEXT


def build_user_message(
    *,
    question: str,
    history: Sequence[ChatMessage],
    facts: Sequence[Fact],
    refs: Mapping[str, ChatMatchRef],
    docs: Mapping[str, DocumentView],
    tool_notes: Sequence[str],
    feedback: str | None,
) -> str:
    """Sections in order, each introduced by a header line: DOCUMENTS IN SCOPE (json.dumps of {ref: {title, level,
    skills: [[skill, cluster], ...]}}, so document strings are JSON-escaped data), RESULT LIST (id, rank, score
    lines), FACTS (one line per fact: id, kind, doc_ids, text), TOOL NOTES (errors from earlier tool calls),
    CONVERSATION (the last 10 messages, at most 6,000 characters in total, oldest dropped first), FEEDBACK (only if
    given), QUESTION. Never includes raw document text."""
    result_lines = [_one_line(f"{r.id}: rank {r.rank}, score {r.score:.3f}") for r in refs.values()]
    fact_lines = [_one_line(f"{f.id} [{f.kind}] ({', '.join(f.doc_ids)}): {f.text}") for f in facts]
    note_lines = [_one_line(note) for note in tool_notes]
    sections = [
        ("DOCUMENTS IN SCOPE", _documents_section(docs)),
        ("RESULT LIST", "\n".join(result_lines) or NONE_TEXT),
        ("FACTS", "\n".join(fact_lines) or NONE_TEXT),
        ("TOOL NOTES", "\n".join(note_lines) or NONE_TEXT),
        ("CONVERSATION", _history_section(history)),
    ]
    if feedback:
        sections.append(("FEEDBACK", feedback))
    sections.append(("QUESTION", question))
    return "\n\n".join(f"{header}\n{body}" for header, body in sections)


def _parse_tool(data: Mapping[str, object]) -> Action:
    tool = data.get("tool")
    if tool not in TOOL_NAMES:
        raise ValueError(f"unknown tool {tool!r}")
    args = data.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    return Action(kind="tool", tool=str(tool), args=args)


NOT_A_TOOL_HINT = "a reply that starts with { must be one tool request; otherwise write plain text"
ELLIPSIS = "…"


def _clip(text: str) -> str:
    if len(text) <= MAX_REPLY_CHARS:
        return text
    return text[: MAX_REPLY_CHARS - len(ELLIPSIS)] + ELLIPSIS


def _parse_tool_request(cleaned: str) -> Action:
    try:
        data = parse_json_object(cleaned)
    except ValueError as e:
        raise ValueError(NOT_A_TOOL_HINT) from e
    if data.get("action") != "tool":
        raise ValueError(NOT_A_TOOL_HINT)
    return _parse_tool(data)


def parse_action(text: str) -> Action:
    """A reply starting with "{" must be a valid tool request (ValueError with a short reason otherwise, fed back to
    the model); blank text is ValueError("empty reply"); anything else is the answer, clipped to MAX_REPLY_CHARS."""
    cleaned = strip_reply_wrappers(text)
    if not cleaned:
        raise ValueError("empty reply")
    if cleaned.startswith("{"):
        return _parse_tool_request(cleaned)
    return Action(kind="answer", text=_clip(cleaned))

"""System prompt loading, user-message building and action parsing for the agent."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from getwork.agent.models import MAX_CLAIMS, PROMPT_VERSION, ChatMatchRef, ChatMessage, Claim, Fact
from getwork.agent.tools import TOOL_NAMES
from getwork.llm.validated import parse_json_object
from getwork.webapp.models import DocumentView

MAX_HISTORY_MESSAGES = 10
MAX_HISTORY_TOTAL_CHARS = 6000
NONE_TEXT = "(none)"


@dataclass(frozen=True)
class Action:
    kind: Literal["tool", "answer"]
    tool: str | None = None
    args: dict[str, object] | None = None
    claims: list[Claim] | None = None
    off_topic: bool = False


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


def _parse_claim(index: int, raw: object) -> Claim:
    if not isinstance(raw, dict):
        raise ValueError(f"claims[{index}] must be an object")
    try:
        return Claim.model_validate(raw)
    except ValidationError as e:
        first = e.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        raise ValueError(f"claims[{index}].{location}: {first['msg']}") from e


def _parse_answer(data: Mapping[str, object]) -> Action:
    off_topic = data.get("off_topic", False)
    if not isinstance(off_topic, bool):
        raise ValueError("off_topic must be true or false")
    if off_topic:
        return Action(kind="answer", claims=[], off_topic=True)
    raw_claims = data.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise ValueError("an answer needs at least one claim unless off_topic is true")
    if len(raw_claims) > MAX_CLAIMS:
        raise ValueError(f"at most {MAX_CLAIMS} claims are allowed")
    claims = [_parse_claim(i, raw) for i, raw in enumerate(raw_claims)]
    return Action(kind="answer", claims=claims, off_topic=False)


def parse_action(text: str) -> Action:
    """parse_json_object, then: "action" is "tool" (tool in TOOL_NAMES, args an object) or "answer" (off_topic
    bool default false; claims a list of objects with text, doc_ids, skills, facts validated as Claim; 1 to
    MAX_CLAIMS claims unless off_topic). ValueError with a short reason otherwise (fed back to the model)."""
    data = parse_json_object(text)
    action = data.get("action")
    if action == "tool":
        return _parse_tool(data)
    if action == "answer":
        return _parse_answer(data)
    raise ValueError('"action" must be "tool" or "answer"' if action else 'missing "action"')

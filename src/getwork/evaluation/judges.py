"""The judge contract: rubric, verdict schema, prompt, reply parsing and the independence rule."""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from getwork.llm.validated import parse_json_object

RUBRIC_DEFAULT_FILE: Path = Path(__file__).parent / "prompts" / "judge_v1.md"
LABELS: tuple[int, ...] = (0, 1, 2)
MAX_RATIONALE_CHARS = 500
SENIORITY_CAP = 1
CUT_MARKER = "[text cut]"
HASH_HEX_CHARS = 12
VERDICT_KEYS = frozenset({"label", "seniority_mismatch", "rationale"})
# No length keywords: the provider rejects them (see structure/schema.py).
VERDICT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "label": {"type": "integer", "enum": [0, 1, 2]},
        "seniority_mismatch": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "seniority_mismatch", "rationale"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Rubric:
    version: str  # the file stem, e.g. "judge_v1"
    text: str
    hash: str  # "sha256:" + first 12 hex of sha256(text.encode("utf-8"))


@dataclass(frozen=True)
class JudgeInfo:
    id: str  # "judge-a" or "judge-b"
    model: str
    family: str  # user-declared model family, e.g. "qwen", "llama"
    backend: str


@dataclass(frozen=True)
class PairText:
    pair_id: str
    resume_text: str
    posting_text: str


@dataclass(frozen=True)
class Verdict:
    label: int  # 0, 1 or 2 as the judge wrote it
    seniority_mismatch: bool
    rationale: str  # at most MAX_RATIONALE_CHARS characters


class JudgeError(Exception):
    """One pair failed; reason is one of rejected, invalid_reply, unavailable."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class JudgeFatalError(Exception):
    """The whole run cannot succeed (bad key, unknown model)."""


class Judge(Protocol):
    @property
    def info(self) -> JudgeInfo: ...

    def judge(self, pair: PairText) -> tuple[Verdict, bool]:
        """(verdict, truncated). Raises JudgeError for this pair or JudgeFatalError for the run."""
        ...


def load_rubric(path: Path | None = None) -> Rubric:
    source = RUBRIC_DEFAULT_FILE if path is None else Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"rubric file not found: {source}") from e
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_HEX_CHARS]
    return Rubric(version=source.stem, text=text, hash=f"sha256:{digest}")


def effective_label(verdict: Verdict) -> int:
    if verdict.seniority_mismatch:
        return min(verdict.label, SENIORITY_CAP)
    return verdict.label


def parse_verdict(text: str) -> Verdict:
    data = parse_json_object(text)
    if set(data) != VERDICT_KEYS:
        raise ValueError('the object must have exactly the keys "label", "seniority_mismatch", '
                         '"rationale"')
    label = data["label"]
    if type(label) is not int or label not in LABELS:
        raise ValueError('"label" must be the integer 0, 1 or 2')
    mismatch = data["seniority_mismatch"]
    if type(mismatch) is not bool:
        raise ValueError('"seniority_mismatch" must be true or false')
    rationale = data["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError('"rationale" must be a non-empty string')
    return Verdict(
        label=label,
        seniority_mismatch=mismatch,
        rationale=rationale.strip()[:MAX_RATIONALE_CHARS],
    )


def _cut(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n" + CUT_MARKER, True


def build_prompt(rubric: Rubric, pair: PairText, max_chars: int) -> tuple[str, str, bool]:
    """(system, user, truncated); the texts carry no pair id, system or direction."""
    resume, cut_resume = _cut(pair.resume_text, max_chars)
    posting, cut_posting = _cut(pair.posting_text, max_chars)
    user = f"RESUME:\n{resume}\n\nPOSTING:\n{posting}"
    return rubric.text, user, cut_resume or cut_posting


def check_independence(infos: Sequence[JudgeInfo], used_models: Collection[str]) -> None:
    if len(infos) != 2:
        raise ValueError(f"exactly two judges are required, got {len(infos)}")
    first, second = infos
    if first.id == second.id:
        raise ValueError(f"the two judges must have different ids, both are {first.id!r}")
    for info in infos:
        if not info.family.strip():
            raise ValueError(f"judge {info.id} has a blank model family")
    if first.family.strip().casefold() == second.family.strip().casefold():
        raise ValueError(f"the two judges must come from different families, both are "
                         f"{first.family!r}")
    used = {m.casefold() for m in used_models}
    for info in infos:
        if info.model.casefold() in used:
            raise ValueError(f"judge {info.id} model {info.model!r} is also used by the pipeline")

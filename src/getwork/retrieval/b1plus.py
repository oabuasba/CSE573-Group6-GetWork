"""B1+ primitives: skill, title and seniority similarity with tunable weights."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path

from getwork.alias import normalize
from getwork.docembed import levels

SENIORITY_UNKNOWN: float = 0.5
TITLE_STOPWORDS: frozenset[str] = frozenset(
    {"a", "an", "and", "at", "for", "in", "of", "or", "the", "to", "with"}
)
WEIGHT_NAMES: tuple[str, ...] = ("skills", "title", "level")
SUM_TOLERANCE = 1e-6
STEP_TOLERANCE = 1e-9
WEIGHT_DECIMALS = 10
_TITLE_TOKEN = re.compile(r"[a-z0-9][a-z0-9#+.]*")


@dataclass(frozen=True)
class B1PlusWeights:
    skills: float
    title: float
    level: float

    def __post_init__(self) -> None:
        values = self.as_dict()
        for name, value in values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"weight {name} must be finite and >= 0, got {value}")
        if abs(sum(values.values()) - 1) > SUM_TOLERANCE:
            raise ValueError(f"weights must sum to 1, got {sum(values.values())}")

    def as_dict(self) -> dict[str, float]:
        return {"skills": self.skills, "title": self.title, "level": self.level}


DEFAULT_WEIGHTS: B1PlusWeights = B1PlusWeights(0.7, 0.15, 0.15)


def parse_weights(text: str) -> B1PlusWeights:
    """Parse "0.7,0.15,0.15" into weights; ValueError unless exactly three numbers."""
    parts = text.split(",")
    if len(parts) != len(WEIGHT_NAMES):
        raise ValueError(f"expected three comma-separated weights, got {text!r}")
    try:
        skills, title, level = (float(p) for p in parts)
    except ValueError as e:
        raise ValueError(f"weights must be numbers, got {text!r}") from e
    return B1PlusWeights(skills=skills, title=title, level=level)


def load_weights(path: Path) -> B1PlusWeights:
    """Read a JSON object with exactly the keys skills, title, level."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"weights file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"weights file {path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict) or set(raw) != set(WEIGHT_NAMES):
        raise ValueError(f"weights file {path} must be an object with keys {WEIGHT_NAMES}")
    try:
        return B1PlusWeights(**{name: raw[name] for name in WEIGHT_NAMES})
    except (TypeError, ValueError) as e:
        raise ValueError(f"weights file {path} has invalid values: {e}") from e


def title_tokens(title: str | None) -> frozenset[str]:
    """Casefolded word tokens of a title without stop words; empty for None or blank."""
    if not title or not title.strip():
        return frozenset()
    tokens = (t.rstrip(".") for t in _TITLE_TOKEN.findall(title.casefold()))
    return frozenset(t for t in tokens if t and t not in TITLE_STOPWORDS)


def jaccard(a: AbstractSet[str], b: AbstractSet[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def skill_keys(skills: Sequence[str]) -> frozenset[str]:
    keys = (normalize.normalize_key(s) for s in skills)
    return frozenset(k for k in keys if k)


def seniority_similarity(level_a: str, level_b: str) -> float:
    rank_a, rank_b = levels.level_rank(level_a), levels.level_rank(level_b)
    if rank_a is None or rank_b is None:
        return SENIORITY_UNKNOWN
    return 1 - abs(rank_a - rank_b) / (len(levels.LEVELS) - 1)


def b1plus_score(
    resume_doc: Mapping[str, object], posting_doc: Mapping[str, object], w: B1PlusWeights
) -> float:
    """Reference score on two documents-table rows ("skills", "title", "level")."""
    skills = w.skills * jaccard(
        skill_keys(list(resume_doc["skills"])), skill_keys(list(posting_doc["skills"]))
    )
    title = w.title * jaccard(
        title_tokens(resume_doc["title"]), title_tokens(posting_doc["title"])
    )
    level = w.level * seniority_similarity(str(resume_doc["level"]), str(posting_doc["level"]))
    return skills + title + level


def grid_search_weights(
    objective: Callable[[B1PlusWeights], float], step: float = 0.05
) -> tuple[B1PlusWeights, float]:
    """Best weights over the simplex grid with the given step; first candidate wins ties."""
    if not 0 < step <= 0.5:
        raise ValueError(f"step must be in (0, 0.5], got {step}")
    n = round(1 / step)
    if abs(1 / step - n) > STEP_TOLERANCE:
        raise ValueError(f"1/step must be an integer, got step {step}")
    best: tuple[B1PlusWeights, float] | None = None
    for i in range(n, -1, -1):
        for j in range(n - i, -1, -1):
            w = B1PlusWeights(
                skills=round(i * step, WEIGHT_DECIMALS),
                title=round(j * step, WEIGHT_DECIMALS),
                level=round((n - i - j) * step, WEIGHT_DECIMALS),
            )
            score = objective(w)
            if best is None or score > best[1]:
                best = (w, score)
    assert best is not None
    return best

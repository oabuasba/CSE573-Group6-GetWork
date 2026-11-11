"""Version families: fold versioned spellings of a reviewed product into one canonical skill."""

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from getwork.alias import normalize
from getwork.alias.table import AliasTable

FAMILIES_PATH: Path = Path(__file__).parent / "version_families.csv"
FAMILIES_HEADER: list[str] = ["base", "canonical", "map_bare"]
TOP_VERSION_FOLDS: int = 20
VERSION_CANDIDATE_LIMIT: int = 50

_NUM = r"\d+(?:\.(?:\d+|x))*[a-z]?"
_NAMED = r"(?:xp|vista|nt|ce)(?:\s?" + _NUM + r")?"
_ONE = r"(?:" + _NAMED + r"|" + _NUM + r")(?:\s?(?:r\d|sp\d|service pack \d|update \d+))?"
_SEP = r"\s*(?:/|,|&|-|and|or)\s*"
_VERSION = _ONE + r"(?:" + _SEP + _ONE + r")*"
_PREFIX = r"[\s_-]*(?:v|ver\.?|version)?[\s_-]*"
_CANDIDATE = re.compile(
    r"(?P<base>.*?\S)[\s_-]*(?:v|ver\.?|version)?[\s_-]*\d+(?:\.\d+)*[a-z]?"
)


@dataclass(frozen=True)
class VersionFamily:
    base: str
    canonical: str
    map_bare: bool = True


def _parse_map_bare(cell: str, line: int) -> bool:
    value = cell.strip().casefold()
    if value in ("", "yes"):
        return True
    if value == "no":
        return False
    raise ValueError(f"line {line}: map_bare {cell!r} must be yes or no")


def _parse_family_row(row: list[str], line: int) -> VersionFamily:
    cells = [cell.strip() for cell in row]
    if len(cells) < 2 or not cells[0] or not cells[1]:
        raise ValueError(f"line {line}: expected non-blank cells base,canonical")
    for name, value in (("base", cells[0]), ("canonical", cells[1])):
        if not normalize.normalize_key(value):
            raise ValueError(f"line {line}: {name} {value!r} normalizes to an empty key")
    map_bare = _parse_map_bare(cells[2], line) if len(cells) > 2 else True
    return VersionFamily(base=cells[0], canonical=cells[1], map_bare=map_bare)


def load_families(path: Path = FAMILIES_PATH) -> list[VersionFamily]:
    families: list[VersionFamily] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or [c.strip() for c in header] != FAMILIES_HEADER:
            raise ValueError("line 1: missing header base,canonical,map_bare")
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            family = _parse_family_row(row, reader.line_num)
            key = normalize.normalize_key(family.base)
            if key in seen:
                raise ValueError(
                    f"line {reader.line_num}: base {family.base!r} repeats an earlier row"
                )
            seen.add(key)
            families.append(family)
    return families


@lru_cache(maxsize=None)
def _family_pattern(base_key: str) -> re.Pattern[str]:
    return re.compile(re.escape(base_key) + _PREFIX + _VERSION)


def fold_version(skill: str, families: Sequence[VersionFamily]) -> str | None:
    key = normalize.normalize_key(skill)
    if not key:
        return None
    best: tuple[int, VersionFamily] | None = None
    for family in families:
        base_key = normalize.normalize_key(family.base)
        if best is not None and len(base_key) <= best[0]:
            continue
        if _family_pattern(base_key).fullmatch(key):
            best = (len(base_key), family)
    return None if best is None else normalize.clean_surface(best[1].canonical)


def _top_folds(
    forms: Mapping[str, int], families: Sequence[VersionFamily]
) -> tuple[int, list[dict]]:
    grouped: dict[str, dict[str, int]] = {}
    for form, count in forms.items():
        canonical = fold_version(form, families)
        if canonical is not None:
            grouped.setdefault(canonical, {})[form] = count
    n_folded = sum(len(group) for group in grouped.values())
    ranked = sorted(grouped.items(), key=lambda item: (-sum(item[1].values()), item[0]))
    top = [{"canonical": c, "forms": sorted(g)} for c, g in ranked[:TOP_VERSION_FOLDS]]
    return n_folded, top


def _version_candidates(
    forms: Mapping[str, int], families: Sequence[VersionFamily], table: AliasTable
) -> list[dict]:
    candidates = []
    for form, count in forms.items():
        key = normalize.normalize_key(form)
        match = _CANDIDATE.fullmatch(key)
        if match is None:
            continue
        base_key = match.group("base")
        if base_key == key or base_key not in table.entries:
            continue
        if fold_version(form, families) is not None:
            continue
        merged = table.canonical(form)
        if merged is None or normalize.normalize_key(merged) != key:
            continue
        candidates.append({"skill": form, "base": table.entries[base_key], "count": count})
    candidates.sort(key=lambda c: (-c["count"], c["skill"]))
    return candidates[:VERSION_CANDIDATE_LIMIT]


def version_report(
    forms: Mapping[str, int], families: Sequence[VersionFamily], table: AliasTable
) -> dict:
    n_folded, top_folds = _top_folds(forms, families)
    return {
        "alias_table_version": table.version,
        "n_families": len(families),
        "n_version_forms_folded": n_folded,
        "top_version_folds": top_folds,
        "version_candidates": _version_candidates(forms, families, table),
    }

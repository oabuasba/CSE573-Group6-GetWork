"""Build the alias table from corpus surface forms plus the curated CSV."""

import csv
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from getwork.alias import normalize
from getwork.alias import table as alias_table
from getwork.alias.table import AliasTable
from getwork.alias.versions import VersionFamily, fold_version

CURATED_PATH: Path = Path(__file__).parent / "alias_curated.csv"
CURATED_HEADER = ["alias", "canonical"]
# An all-caps form with 5 or more letters (JAVASCRIPT) is shouting; JAVA, AWS, HTML are acronyms.
SHOUTING_MIN_LETTERS = 5
TOP_MERGES = 20


@dataclass(frozen=True)
class CuratedPair:
    alias: str
    canonical: str


def _parse_curated_row(row: list[str], line: int) -> CuratedPair:
    cells = [cell.strip() for cell in row]
    if len(cells) < 2 or not cells[0] or not cells[1]:
        raise ValueError(f"line {line}: expected two non-blank cells alias,canonical")
    pair = CuratedPair(alias=cells[0], canonical=cells[1])
    for name, value in (("alias", pair.alias), ("canonical", pair.canonical)):
        if not normalize.normalize_key(value):
            raise ValueError(f"line {line}: {name} {value!r} normalizes to an empty key")
    return pair


def load_curated(path: Path = CURATED_PATH) -> list[CuratedPair]:
    pairs: list[CuratedPair] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None or [c.strip() for c in header] != CURATED_HEADER:
            raise ValueError("line 1: missing header alias,canonical")
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            pairs.append(_parse_curated_row(row, reader.line_num))
    return pairs


def count_surface_forms(skill_lists: Iterable[Sequence[str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for skills in skill_lists:
        forms = {normalize.clean_surface(s) for s in skills}
        counts.update(form for form in forms if form)
    return counts


def _is_shouting(form: str) -> bool:
    letters = sum(1 for ch in form if ch.isalpha())
    return form.isupper() and letters >= SHOUTING_MIN_LETTERS


def pick_display(forms: Mapping[str, int]) -> str:
    if not forms:
        raise ValueError("no forms to pick from")
    candidates = [f for f in forms if not _is_shouting(f)] or list(forms)
    return sorted(candidates, key=lambda f: (-forms[f], f.islower(), f))[0]


def _check_pair(entries: Mapping[str, str], pair: CuratedPair) -> None:
    canonical = normalize.clean_surface(pair.canonical)
    alias_ok = entries[normalize.normalize_key(pair.alias)] == canonical
    self_ok = entries[normalize.normalize_key(pair.canonical)] == canonical
    if not (alias_ok and self_ok):
        raise ValueError(
            f"conflicting curated row: {pair.alias!r} -> {pair.canonical!r} "
            "(alias has two canonicals, or the canonical is itself an alias)"
        )


def _check_family(entries: Mapping[str, str], family: VersionFamily) -> None:
    canonical = normalize.clean_surface(family.canonical)
    keys = [normalize.normalize_key(family.canonical)]
    if family.map_bare:
        keys.append(normalize.normalize_key(family.base))
    if any(entries[key] != canonical for key in keys):
        raise ValueError(
            f"conflicting version family: {family.base!r} -> {family.canonical!r} "
            "(the canonical is itself folded or aliased to something else)"
        )


def _apply_families(entries: dict[str, str], families: Sequence[VersionFamily]) -> None:
    for family in families:
        canonical = normalize.clean_surface(family.canonical)
        entries[normalize.normalize_key(family.canonical)] = canonical
        if family.map_bare:
            entries[normalize.normalize_key(family.base)] = canonical
    for key in list(entries):
        folded = fold_version(key, families)
        if folded is not None:
            entries[key] = folded


def build_table(
    forms: Mapping[str, int],
    curated: Sequence[CuratedPair],
    families: Sequence[VersionFamily] = (),
) -> AliasTable:
    groups: dict[str, dict[str, int]] = defaultdict(dict)
    for form, count in forms.items():
        key = normalize.normalize_key(form)
        if key:
            groups[key][form] = count
    entries = {key: pick_display(group) for key, group in groups.items()}
    _apply_families(entries, families)
    for pair in curated:
        canonical = normalize.clean_surface(pair.canonical)
        entries[normalize.normalize_key(pair.canonical)] = canonical
        entries[normalize.normalize_key(pair.alias)] = canonical
    for pair in curated:
        _check_pair(entries, pair)
    for family in families:
        _check_family(entries, family)
    return alias_table.make_table(entries)


def _group_by_canonical(forms: Mapping[str, int], table: AliasTable) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for form, count in forms.items():
        canonical = table.canonical(form)
        if canonical is not None:
            grouped[canonical][form] = count
    return grouped


def canonical_counts(forms: Mapping[str, int], table: AliasTable) -> list[tuple[str, int]]:
    totals = {c: sum(g.values()) for c, g in _group_by_canonical(forms, table).items()}
    return sorted(totals.items(), key=lambda kv: (-kv[1], kv[0].casefold(), kv[0]))


def build_report(forms: Mapping[str, int], table: AliasTable, n_curated: int) -> dict:
    grouped = _group_by_canonical(forms, table)
    merged = sorted(
        ((c, g) for c, g in grouped.items() if len(g) >= 2),
        key=lambda item: (-sum(item[1].values()), item[0]),
    )
    return {
        "alias_table_version": table.version,
        "n_keys": len(table.entries),
        "n_raw_forms": len(forms),
        "n_curated_pairs": n_curated,
        "n_forms_merged": len(forms) - len(grouped),
        "top_merges": [
            {"canonical": c, "forms": sorted(g)} for c, g in merged[:TOP_MERGES]
        ],
    }

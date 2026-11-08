"""The alias table: normalized key to canonical spelling, with a content version."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from getwork import fileio
from getwork.alias import normalize

VERSION_PREFIX = "sha256:"
VERSION_HEX_CHARS = 12


def table_version(entries: Mapping[str, str]) -> str:
    payload = json.dumps(entries, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return VERSION_PREFIX + hashlib.sha256(payload).hexdigest()[:VERSION_HEX_CHARS]


@dataclass(frozen=True)
class AliasTable:
    entries: Mapping[str, str]
    version: str

    def canonical(self, skill: str) -> str | None:
        key = normalize.normalize_key(skill)
        if not key:
            return None
        return self.entries.get(key, normalize.clean_surface(skill))

    def canonicalize_list(self, skills: Sequence[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for skill in skills:
            canonical = self.canonical(skill)
            if canonical is None:
                continue
            key = normalize.normalize_key(canonical)
            if key in seen:
                continue
            seen.add(key)
            result.append(canonical)
        return result


def make_table(entries: Mapping[str, str]) -> AliasTable:
    return AliasTable(dict(entries), table_version(entries))


def save_table(path: Path, table: AliasTable, meta: Mapping[str, object]) -> None:
    fileio.write_json_atomic(path, {
        "version": table.version,
        "n_entries": len(table.entries),
        "meta": dict(meta),
        "entries": dict(table.entries),
    })


def load_table(path: Path) -> AliasTable:
    data = fileio.read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
        raise ValueError(f"{path}: not an alias table file")
    table = make_table(data["entries"])
    if data.get("version") != table.version:
        raise ValueError(
            f"{path}: stored version {data.get('version')!r} does not match entries "
            f"({table.version}); the file was edited"
        )
    return table

"""Resumable answer cache stamped with the taxonomy, model, backend and prompt version."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from getwork import fileio

PROVENANCE_FIELDS = ("model", "backend", "prompt_version")


class RelabelProvenanceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Stamp:
    taxonomy_version: str
    model: str
    backend: str
    prompt_version: str


class StampedCache:
    ignored: int  # entries dropped at load because their taxonomy_version differs

    def __init__(self, path: Path, stamp: Stamp, *, allow_mixed: bool = False) -> None:
        """Load path if it exists; drop other-taxonomy entries, reject other-provenance ones.

        A corrupt file raises ValueError naming the path. Keys are never put in messages.
        """
        self._path = path
        self._stamp = stamp
        self._entries: dict[str, dict] = {}
        self.ignored = 0
        if path.exists():
            self._load(allow_mixed=allow_mixed)

    def _load(self, *, allow_mixed: bool) -> None:
        entries = self._read_entries()
        differing: dict[str, set[str]] = {}
        for key, entry in entries.items():
            if entry.get("taxonomy_version") != self._stamp.taxonomy_version:
                self.ignored += 1
                continue
            for name in PROVENANCE_FIELDS:
                if entry.get(name) != getattr(self._stamp, name):
                    differing.setdefault(name, set()).add(str(entry.get(name)))
            self._entries[key] = entry
        if differing and not allow_mixed:
            raise RelabelProvenanceError(self._provenance_message(differing))

    def _read_entries(self) -> dict[str, dict]:
        try:
            raw = fileio.read_json(self._path)
        except ValueError as e:
            raise ValueError(f"corrupt relabel cache: {self._path}") from e
        entries = raw.get("entries") if isinstance(raw, dict) else None
        if not isinstance(entries, dict) or not all(
            isinstance(v, dict) and "value" in v for v in entries.values()
        ):
            raise ValueError(f"corrupt relabel cache: {self._path}")
        return entries

    def _provenance_message(self, differing: dict[str, set[str]]) -> str:
        parts = [
            f"{name}: cache has {', '.join(sorted(differing[name]))!r}, "
            f"this run uses {getattr(self._stamp, name)!r}"
            for name in PROVENANCE_FIELDS
            if name in differing
        ]
        return (
            f"{self._path} was written with a different setup ({'; '.join(parts)}); "
            "use a fresh cache directory or pass --allow-mixed"
        )

    def get(self, key: str) -> object | None:
        entry = self._entries.get(key)
        return None if entry is None else entry["value"]

    def put(self, key: str, value: object) -> None:
        self._entries[key] = {
            "value": value,
            "taxonomy_version": self._stamp.taxonomy_version,
            "model": self._stamp.model,
            "backend": self._stamp.backend,
            "prompt_version": self._stamp.prompt_version,
        }

    def save(self) -> None:
        fileio.write_json_atomic(self._path, {"entries": self._entries})

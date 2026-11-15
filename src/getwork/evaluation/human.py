"""Read and validate the author label files (human/<author>.jsonl)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from getwork.evaluation import layout

AUTHOR_SLUG = layout.AUTHOR_SLUG
HUMAN_STAGE = "human"
HUMAN_DIR = "human"
LABELS = (0, 1, 2)


def list_authors(out_dir: Path) -> list[str]:
    folder = out_dir / HUMAN_DIR
    if not folder.is_dir():
        return []
    return sorted(
        path.stem
        for path in folder.glob("*.jsonl")
        if AUTHOR_SLUG.fullmatch(path.stem) and path.stem != layout.RESERVED_AUTHOR
    )


def _json_lines(path: Path) -> Iterator[tuple[int, object]]:
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield number, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path} line {number}: invalid JSON ({e.msg})") from e


def _blind_pair_ids(out_dir: Path) -> set[str]:
    path = layout.blind_path(out_dir, HUMAN_STAGE)
    if not path.exists():
        raise FileNotFoundError(f"human blind file not found: {path}")
    ids = set()
    for number, row in _json_lines(path):
        if not isinstance(row, dict) or not isinstance(row.get("pair_id"), str):
            raise ValueError(f"{path} line {number}: row has no pair_id")
        ids.add(row["pair_id"])
    return ids


def _check_row(row: object, *, where: str, author: str, allowed: set[str]) -> tuple[str, int]:
    if not isinstance(row, dict):
        raise ValueError(f"{where}: row is not a JSON object")
    if row.get("author") != author:
        raise ValueError(
            f"{where}: author {row.get('author')!r} differs from the file's {author!r}"
        )
    label = row.get("label")
    if isinstance(label, bool) or not isinstance(label, int) or label not in LABELS:
        raise ValueError(f"{where}: label {label!r} is not an integer in {list(LABELS)}")
    pair_id = row.get("pair_id")
    if pair_id not in allowed:
        raise ValueError(f"{where}: pair_id {pair_id!r} is not in the human blind file")
    return pair_id, label


def read_human_labels(out_dir: Path, author: str) -> dict[str, int]:
    """pair_id -> label for one author; a later row for a pair replaces an earlier one."""
    path = layout.human_path(out_dir, author)
    if not path.exists():
        raise FileNotFoundError(f"author label file not found: {path}")
    allowed = _blind_pair_ids(out_dir)
    labels: dict[str, int] = {}
    for number, row in _json_lines(path):
        pair_id, label = _check_row(
            row, where=f"{path} line {number}", author=author, allowed=allowed
        )
        labels[pair_id] = label
    return labels

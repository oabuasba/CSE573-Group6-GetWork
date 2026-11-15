from __future__ import annotations

import json
from pathlib import Path

import pytest

from getwork.evaluation import human

pytestmark = pytest.mark.usefixtures("offline")

PAIRS = ["p-aaa", "p-bbb", "p-ccc"]


def _write(path: Path, rows: list[object], *, raw: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = raw if raw is not None else [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _setup(out: Path) -> None:
    rows = [{"pair_id": p, "resume": "r", "posting": "q"} for p in PAIRS]
    _write(out / "pairs_blind_human.jsonl", rows)


def _row(pair_id: str, label: object, author: str = "omar") -> dict:
    return {"pair_id": pair_id, "author": author, "label": label, "labeled_at": "t"}


def test_read_human_labels(tmp_path: Path) -> None:
    _setup(tmp_path)
    rows = [_row("p-aaa", 0), _row("p-bbb", 1), _row("p-aaa", 2)]
    raw = [json.dumps(r) for r in rows[:2]] + ["", json.dumps(rows[2])]
    _write(tmp_path / "human" / "omar.jsonl", [], raw=raw)
    assert human.read_human_labels(tmp_path, "omar") == {"p-aaa": 2, "p-bbb": 1}


def test_read_human_labels_rejects_bad_rows(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pairs_blind_human"):
        _write(tmp_path / "human" / "omar.jsonl", [_row("p-aaa", 1)])
        human.read_human_labels(tmp_path, "omar")
    _setup(tmp_path)
    with pytest.raises(FileNotFoundError, match="nobody.jsonl"):
        human.read_human_labels(tmp_path, "nobody")
    path = tmp_path / "human" / "omar.jsonl"
    cases = {
        "invalid JSON": ([], ["{not json"]),
        "label": ([_row("p-aaa", 3)], None),
        "label True": ([_row("p-aaa", True)], None),
        "author": ([_row("p-aaa", 1, author="other")], None),
        "pair_id": ([_row("p-zzz", 1)], None),
    }
    for needle, (rows, raw) in cases.items():
        _write(path, rows, raw=raw)
        with pytest.raises(ValueError, match=r"omar\.jsonl line 1") as info:
            human.read_human_labels(tmp_path, "omar")
        assert needle.split()[0] in str(info.value)


def test_list_authors(tmp_path: Path) -> None:
    assert human.list_authors(tmp_path) == []
    for name in ("zed", "omar", "judge", "Bad Name", "x" * 21):
        _write(tmp_path / "human" / f"{name}.jsonl", [])
    (tmp_path / "human" / "notes.txt").write_text("x")
    assert human.list_authors(tmp_path) == ["omar", "zed"]

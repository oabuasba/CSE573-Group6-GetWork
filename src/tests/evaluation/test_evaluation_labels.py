from __future__ import annotations

from pathlib import Path

import pytest

from getwork import fileio
from getwork.evaluation import labels

pytestmark = pytest.mark.usefixtures("offline")


def _row(label: int, status: str = "ok") -> dict:
    return {"status": status, "effective_label": label}


def test_combine_labels() -> None:
    assert labels.combine_labels(2, 0) == 0
    assert labels.combine_labels(1, 2) == 1
    assert labels.combine_labels(2, 2) == 2
    assert labels.combine_labels(None, 1) is None
    assert labels.combine_labels(1, None) is None
    assert labels.combine_labels(None, None) is None


def _tables() -> tuple[dict, dict]:
    rows_a = {"p-3": _row(2), "p-1": _row(1), "p-2": _row(2), "p-4": _row(0, "failed")}
    rows_b = {"p-1": _row(1), "p-3": _row(0), "p-2": _row(1), "p-5": _row(2)}
    return rows_a, rows_b


def test_build_labels_table_lower_wins_with_cap() -> None:
    rows_a, rows_b = _tables()
    rows_a["p-2"] = _row(1)  # a judge's 2 with seniority mismatch arrives already capped to 1
    table = labels.build_labels_table(rows_a, rows_b)
    assert table.schema == labels.LABELS_SCHEMA
    assert table.to_pydict() == {
        "pair_id": ["p-1", "p-2", "p-3"],
        "label_a": [1, 1, 2],
        "label_b": [1, 1, 0],
        "label": [1, 1, 0],
        "disagree": [False, False, True],
    }
    empty = labels.build_labels_table({}, {})
    assert empty.schema == labels.LABELS_SCHEMA and empty.num_rows == 0


def test_read_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.parquet"
    fileio.write_parquet_atomic(path, labels.build_labels_table(*_tables()))
    assert labels.read_labels(path) == {"p-1": 1, "p-2": 1, "p-3": 0}
    with pytest.raises(FileNotFoundError, match="missing.parquet"):
        labels.read_labels(tmp_path / "missing.parquet")

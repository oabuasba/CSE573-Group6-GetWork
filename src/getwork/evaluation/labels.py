"""The combined label table: the lower of the two judges' effective labels per pair."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LABELS_SCHEMA: pa.Schema = pa.schema(
    [
        ("pair_id", pa.string()),
        ("label_a", pa.int8()),
        ("label_b", pa.int8()),
        ("label", pa.int8()),
        ("disagree", pa.bool_()),
    ]
)
OK_STATUS = "ok"


def combine_labels(a: int | None, b: int | None) -> int | None:
    if a is None or b is None:
        return None
    return min(a, b)


def build_labels_table(rows_a: Mapping[str, dict], rows_b: Mapping[str, dict]) -> pa.Table:
    ok_a = {k for k, row in rows_a.items() if row["status"] == OK_STATUS}
    ok_b = {k for k, row in rows_b.items() if row["status"] == OK_STATUS}
    pair_ids = sorted(ok_a & ok_b)
    label_a = [rows_a[p]["effective_label"] for p in pair_ids]
    label_b = [rows_b[p]["effective_label"] for p in pair_ids]
    return pa.table(
        {
            "pair_id": pair_ids,
            "label_a": label_a,
            "label_b": label_b,
            "label": [min(a, b) for a, b in zip(label_a, label_b)],
            "disagree": [a != b for a, b in zip(label_a, label_b)],
        },
        schema=LABELS_SCHEMA,
    )


def read_labels(path: Path) -> dict[str, int]:
    try:
        table = pq.read_table(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"labels file not found: {path}") from e
    return dict(zip(table.column("pair_id").to_pylist(), table.column("label").to_pylist()))

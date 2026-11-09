"""Grid, context, selection and review files (ADR-6: the chosen configuration is an artifact)."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from getwork import fileio
from getwork.taxonomy.grid import ConfigResult

GRID_COLUMNS: tuple[str, ...] = (
    "config_id",
    "method",
    "params",
    "n_clusters",
    "noise_fraction",
    "silhouette",
    "largest_cluster_fraction",
    "skipped",
)
REVIEW_COLUMNS: tuple[str, ...] = ("cluster_id", "size", "df_sum", "members")
NOISE_CLUSTER_ID = -1
GRID_RESULTS_NAME = "grid_results.csv"
GRID_LABELS_NAME = "grid_labels.npy"
CONTEXT_NAME = "grid_context.json"
SELECTION_NAME = "selection.json"


def _missing(path: Path) -> FileNotFoundError:
    return FileNotFoundError(f"file not found: {path} (run taxonomy grid)")


def _blank_if_none(value: object) -> object:
    return "" if value is None else value


def _result_row(result: ConfigResult) -> list[object]:
    return [
        result.config_id,
        result.method,
        json.dumps(dict(result.params), sort_keys=True),
        result.n_clusters,
        result.noise_fraction,
        _blank_if_none(result.silhouette),
        result.largest_cluster_fraction,
        _blank_if_none(result.skipped),
    ]


def write_grid(directory: Path, results: Sequence[ConfigResult], labels: np.ndarray) -> None:
    fileio.write_csv_atomic(
        directory / GRID_RESULTS_NAME, GRID_COLUMNS, [_result_row(r) for r in results]
    )
    matrix = np.asarray(labels, dtype=np.int32)

    def write_npy(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            np.save(f, matrix)

    fileio.atomic_write(directory / GRID_LABELS_NAME, write_npy)


def _result_from_row(row: Mapping[str, str]) -> ConfigResult:
    return ConfigResult(
        config_id=row["config_id"],
        method=row["method"],
        params=json.loads(row["params"]),
        n_clusters=int(row["n_clusters"]),
        noise_fraction=float(row["noise_fraction"]),
        silhouette=float(row["silhouette"]) if row["silhouette"] != "" else None,
        largest_cluster_fraction=float(row["largest_cluster_fraction"]),
        skipped=row["skipped"] if row["skipped"] != "" else None,
    )


def read_grid(directory: Path) -> tuple[list[ConfigResult], np.ndarray]:
    results_path = directory / GRID_RESULTS_NAME
    labels_path = directory / GRID_LABELS_NAME
    for path in (results_path, labels_path):
        if not path.exists():
            raise _missing(path)
    with open(results_path, encoding="utf-8", newline="") as f:
        results = [_result_from_row(row) for row in csv.DictReader(f)]
    return results, np.load(labels_path)


def _write_named_json(directory: Path, name: str, payload: Mapping[str, object]) -> None:
    fileio.write_json_atomic(directory / name, dict(payload))


def _read_named_json(directory: Path, name: str) -> dict:
    path = directory / name
    if not path.exists():
        raise _missing(path)
    return fileio.read_json(path)


def write_context(directory: Path, context: Mapping[str, object]) -> None:
    _write_named_json(directory, CONTEXT_NAME, context)


def read_context(directory: Path) -> dict:
    return _read_named_json(directory, CONTEXT_NAME)


def write_selection(directory: Path, record: Mapping[str, object]) -> None:
    _write_named_json(directory, SELECTION_NAME, record)


def read_selection(directory: Path) -> dict:
    return _read_named_json(directory, SELECTION_NAME)


def _review_row(
    cluster_id: int,
    indices: np.ndarray,
    *,
    skills: Sequence[str],
    df: Sequence[int],
    top: int,
) -> dict:
    return {
        "cluster_id": cluster_id,
        "size": int(len(indices)),
        "df_sum": int(sum(df[i] for i in indices)),
        "members": "; ".join(skills[i] for i in indices[:top]),
    }


def review_rows(
    labels: np.ndarray, skills: Sequence[str], df: Sequence[int], top_members: int = 25
) -> list[dict]:
    labels = np.asarray(labels)
    ids = sorted(int(c) for c in np.unique(labels) if c >= 0)
    if (labels == NOISE_CLUSTER_ID).any():
        ids.append(NOISE_CLUSTER_ID)
    return [
        _review_row(c, np.flatnonzero(labels == c), skills=skills, df=df, top=top_members)
        for c in ids
    ]


def write_review(directory: Path, config_id: str, rows: Sequence[Mapping[str, object]]) -> Path:
    path = directory / f"review_{config_id}.csv"
    body = [[row[c] for c in REVIEW_COLUMNS] for row in rows]
    fileio.write_csv_atomic(path, REVIEW_COLUMNS, body)
    return path

"""Read the documents tables, first occupation labels, texts and pipeline model names."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from getwork import paths
from getwork.docembed import documents
from getwork.retrieval import inputs
from getwork.webapp import artifacts

MAX_MISSING_SHOWN = 5


@dataclass(frozen=True)
class EvalInputs:
    root: Path
    data: inputs.RetrievalInputs
    first_labels: dict[str, str | None]


def load_eval_inputs(artifacts_dir: Path, processed_dir: Path) -> EvalInputs:
    root = artifacts.resolve_artifacts_root(artifacts_dir)
    data = inputs.load_inputs(root)
    return EvalInputs(
        root=root, data=data, first_labels=read_first_labels(processed_dir, list(data.resumes))
    )


def read_first_labels(processed_dir: Path, ids: Sequence[str]) -> dict[str, str | None]:
    """id -> first occupation label (None when null or empty), in the order of ids."""
    table = pq.read_table(
        paths.processed_parquet(processed_dir, "resumes"), columns=["id", "first_label"]
    )
    by_id = dict(zip(table.column("id").to_pylist(), table.column("first_label").to_pylist()))
    for id_ in ids:
        if id_ not in by_id:
            raise ValueError(f"resume {id_!r} is not in the processed resumes table")
    return {id_: by_id[id_] or None for id_ in ids}


def read_texts(processed_dir: Path, source: str, ids: Sequence[str]) -> dict[str, str]:
    table = pq.read_table(
        paths.processed_parquet(processed_dir, source), columns=["id", "text_clean"]
    )
    by_id = dict(zip(table.column("id").to_pylist(), table.column("text_clean").to_pylist()))
    missing = [id_ for id_ in ids if id_ not in by_id]
    if missing:
        shown = missing[:MAX_MISSING_SHOWN]
        raise KeyError(f"no text for {source} ids {shown} ({len(missing)} missing)")
    return {id_: by_id[id_] for id_ in ids}


def _canonical_models(root: Path, source: str) -> set[str]:
    path = paths.canonical_dir(root, source) / f"{source}_canonical.jsonl"
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {m for line in f if line.strip() and (m := json.loads(line).get("model"))}


def pipeline_models(root: Path, extra: Sequence[str] = ()) -> list[str]:
    """Sorted distinct non-empty models: canonical records, relabel summaries and extra."""
    models = set(extra)
    for source in paths.SOURCES:
        models |= _canonical_models(root, source)
        relabel_model = documents.read_documents_summary(root=root, source=source).get(
            "relabel_model"
        )
        if relabel_model:
            models.add(relabel_model)
    return sorted(m for m in models if m)

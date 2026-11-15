"""A complete synthetic artifacts and processed tree shared by the evaluation tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

RETRIEVAL_TESTS = Path(__file__).resolve().parents[1] / "retrieval"
if str(RETRIEVAL_TESTS) not in sys.path:
    sys.path.append(str(RETRIEVAL_TESTS))

from retrieval_helpers import SKILLS, make_chain  # noqa: E402

from getwork import fileio, paths  # noqa: E402
from getwork.docembed import documents  # noqa: E402
from getwork.taxonomy.model import save_taxonomy  # noqa: E402

FIRST_LABELS: tuple[str, ...] = ("Data_Role", "Cloud_Role", "Web_Role")
PIPELINE_MODEL = "pipeline-model"
UNLABELED_EVERY = 10
SELECTION = {
    "config_id": "cfg-1", "n_clusters": 4, "silhouette": 0.5, "noise_fraction": 0.1,
}


def _write_canonical(root: Path, source: str, ids: list[str]) -> None:
    rows = [{"id": id_, "model": PIPELINE_MODEL} for id_ in ids]
    fileio.write_jsonl_atomic(
        paths.canonical_dir(root, source) / f"{source}_canonical.jsonl", rows
    )


def _write_processed(processed_dir: Path, source: str, ids: list[str]) -> None:
    columns = {"id": ids, "text_clean": [f"{source[:-1]} text {i} python sql" for i in ids]}
    if source == "resumes":
        columns["first_label"] = [
            None if n % UNLABELED_EVERY == UNLABELED_EVERY - 1 else FIRST_LABELS[n % 3]
            for n in range(len(ids))
        ]
    path = paths.processed_parquet(processed_dir, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path)


def _write_skill_counts(root: Path) -> None:
    rows = [
        {"skill": s, "df_total": 2, "df_resumes": 1, "df_postings": 1, "selected": True, "rank": i}
        for i, s in enumerate(SKILLS, start=1)
    ]
    out = paths.skills_dir(root)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out / "skill_counts.parquet")


def make_eval_tree(
    tmp_path: Path, n_resumes: int = 40, n_postings: int = 30
) -> tuple[Path, Path]:
    """Returns (artifacts_root, processed_dir); ids are r-1.. and p-1.. as in make_chain."""
    root, processed_dir = tmp_path / "art", tmp_path / "processed"
    tax = make_chain(root=root, n_resumes=n_resumes, n_postings=n_postings)
    (root / "alias").mkdir(exist_ok=True)
    save_taxonomy(paths.taxonomy_dir(root) / "taxonomy.json", tax)
    fileio.write_json_atomic(paths.taxonomy_dir(root) / "selection.json", SELECTION)
    _write_skill_counts(root)
    for source in paths.SOURCES:
        ids = list(documents.read_documents(root=root, source=source))
        _write_canonical(root, source, ids)
        _write_processed(processed_dir, source, ids)
    return root, processed_dir

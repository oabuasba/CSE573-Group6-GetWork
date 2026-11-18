"""File names, path helpers, the output-directory guard and the run manifest."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from getwork import fileio
from getwork.docembed.documents import COMPARE_KEYS
from getwork.evaluation import defaults
from getwork.retrieval.inputs import RetrievalInputs

DEFAULT_OUT_DIR: Path = Path("eval/run")
STAGES: tuple[str, ...] = ("dev", "test", "tune")
BLIND_STAGES: tuple[str, ...] = ("dev", "test", "tune", "human")
MANIFEST_FILE = "run.json"
SETS_FILE = "sets.json"
POOL_POSTINGS_FILE = "pool_postings.json"
RUBRIC_FROZEN_FILE = "rubric_frozen.json"
LABELS_JUDGE_FILE = "labels_judge.parquet"
METRICS_FILE = "metrics.json"
AGREEMENT_FILE = "agreement.json"
ARI_FILE = "ari.json"
REPORT_FILE = "report.md"
PROBES_DIR = "probes"
AUTHOR_SLUG = re.compile(r"[a-z0-9_-]{1,20}")
RESERVED_AUTHOR = "judge"
RUN_HINT = "run evaluate sample"


class StaleRunError(ValueError):
    """The artifacts changed after the run was sampled."""


def _check_choice(name: str, value: str, allowed: Sequence[str]) -> str:
    if value not in allowed:
        raise ValueError(f"unknown {name} {value!r}; choose from {list(allowed)}")
    return value


def rankings_path(out_dir: Path, stage: str) -> Path:
    return out_dir / f"pool_rankings_{_check_choice('stage', stage, STAGES)}.parquet"


def pairs_path(out_dir: Path, stage: str) -> Path:
    return out_dir / f"pairs_{_check_choice('stage', stage, STAGES)}.parquet"


def blind_path(out_dir: Path, stage: str) -> Path:
    return out_dir / f"pairs_blind_{_check_choice('stage', stage, BLIND_STAGES)}.jsonl"


def judge_path(out_dir: Path, judge_id: str) -> Path:
    _check_choice("judge id", judge_id, defaults.DEFAULT_JUDGE_IDS)
    return out_dir / "judges" / f"{judge_id}.jsonl"


def human_path(out_dir: Path, author: str) -> Path:
    if not AUTHOR_SLUG.fullmatch(author) or author == RESERVED_AUTHOR:
        raise ValueError(
            f"bad author {author!r}: use 1 to 20 characters of a-z, 0-9, _ or -, "
            f"not {RESERVED_AUTHOR!r}"
        )
    return out_dir / "human" / f"{author}.jsonl"


def tune_dir(out_dir: Path) -> Path:
    return out_dir / "tune"


def check_out_dir(out_dir: Path, protected: Sequence[Path]) -> Path:
    """The resolved out_dir; ValueError when it is, or lies inside, a protected path."""
    resolved = out_dir.resolve()
    for path in protected:
        root = path.resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError(f"--out-dir {out_dir} is inside the protected path {path}")
    return resolved


def stamps_of(data: RetrievalInputs) -> dict:
    return {
        "documents": {k: data.summaries["resumes"].get(k) for k in COMPARE_KEYS},
        "embedding": data.vectors[("resumes", "Full")].meta["embedding"],
        "n_resumes": len(data.resumes),
        "n_postings": len(data.postings),
    }


def check_stamps(manifest: Mapping[str, object], current: Mapping[str, object]) -> None:
    """StaleRunError naming the first stamp that differs from the sampled run."""
    stored = manifest.get("stamps")
    if not isinstance(stored, Mapping):
        raise StaleRunError(f"{MANIFEST_FILE} has no stamps; {RUN_HINT} again")
    for key, value in current.items():
        if stored.get(key) == value:
            continue
        detail = _inner_difference(stored.get(key), value) if key == "documents" else key
        raise StaleRunError(
            f"the artifacts changed since sampling: {detail} differs; "
            f"{RUN_HINT} --force again or restore them"
        )
    extra = [key for key in stored if key not in current]
    if extra:
        raise StaleRunError(f"the artifacts changed since sampling: {extra[0]} differs")


def _inner_difference(stored: object, current: Mapping[str, object]) -> str:
    stored_map = stored if isinstance(stored, Mapping) else {}
    for key, value in current.items():
        if stored_map.get(key) != value:
            return f"documents.{key}"
    return "documents"


def read_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_FILE
    if not path.exists():
        raise FileNotFoundError(f"run manifest not found: {path} ({RUN_HINT})")
    manifest = fileio.read_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} is not a JSON object")
    return manifest


def update_manifest(out_dir: Path, changes: Mapping[str, object]) -> dict:
    path = out_dir / MANIFEST_FILE
    manifest = read_manifest(out_dir) if path.exists() else {}
    merged = {**manifest, **changes}
    fileio.write_json_atomic(path, merged)
    return merged

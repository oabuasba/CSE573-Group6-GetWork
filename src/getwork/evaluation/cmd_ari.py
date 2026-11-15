"""evaluate ari: do cluster profiles recover occupation labels better than raw skills."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from getwork import fileio, paths
from getwork.docembed import documents
from getwork.evaluation import args as eval_args
from getwork.evaluation import ari, loading
from getwork.evaluation.args import EXIT_OK, Context
from getwork.taxonomy import model as taxonomy_model
from getwork.webapp import artifacts

COMMANDS = ("ari",)
DEFAULT_K = 10
DEFAULT_SEEDS = 5
ARI_FILE = "ari.json"
SELECTION_KEYS = ("config_id", "n_clusters", "silhouette", "noise_fraction")


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return number


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("ari", help="adjusted Rand index of cluster versus raw skill profiles")
    eval_args.add_common_args(parser)
    parser.add_argument("--k", type=_positive_int, default=DEFAULT_K)
    parser.add_argument("--seeds", type=_positive_int, default=DEFAULT_SEEDS)


def _require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (run {stage})")
    return path


def _selected_skills(counts: pa.Table) -> list[str]:
    selected = sorted((r for r in counts.to_pylist() if r["selected"]), key=lambda r: r["rank"])
    return [r["skill"] for r in selected]


def run(args: argparse.Namespace, ctx: Context) -> int:
    root = artifacts.resolve_artifacts_root(args.artifacts_dir)
    selection_path = _require(
        paths.taxonomy_dir(root) / "selection.json", "taxonomy select"
    )
    counts_path = _require(
        paths.skills_dir(root) / "skill_counts.parquet", "taxonomy counts"
    )
    docs = documents.read_documents(root, "resumes")
    tax = taxonomy_model.load_taxonomy(paths.taxonomy_dir(root) / "taxonomy.json")
    first_labels = loading.read_first_labels(args.processed_dir, list(docs))
    kept = {id_: doc for id_, doc in docs.items() if first_labels[id_]}
    labels = [first_labels[id_] for id_ in kept]
    counts = pq.read_table(counts_path)
    skills = _selected_skills(counts)
    selection = fileio.read_json(selection_path)
    seeds = list(range(args.seeds))
    result = {
        "schema": 1, "k": args.k, "seeds": seeds, "n_resumes": len(kept),
        "n_unlabeled_excluded": len(docs) - len(kept), "labels": sorted(set(labels)),
        "clusters": ari.ari_over_seeds(
            ari.cluster_profiles(kept, tax.cluster_names()), labels, k=args.k, seeds=seeds
        ),
        "raw_skills": ari.ari_over_seeds(
            ari.skill_profiles(kept, skills), labels, k=args.k, seeds=seeds
        ),
        "taxonomy": {key: selection[key] for key in SELECTION_KEYS},
        "vocabulary_gap": ari.vocabulary_gap(counts),
    }
    fileio.write_json_atomic(args.out_dir / ARI_FILE, result)
    print(json.dumps({
        "clusters": result["clusters"]["ari_mean"], "raw_skills": result["raw_skills"]["ari_mean"],
    }, sort_keys=True))
    return EXIT_OK

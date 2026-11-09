"""The grid and select subcommands (TX-3 to TX-5)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from getwork import paths
from getwork.embed import store
from getwork.taxonomy import cli_common, counts, grid, report
from getwork.taxonomy import select as selection
from getwork.taxonomy.reduce import MIN_ROWS, ReduceConfig, ReduceFn

DEFAULT_MAX_NOISE = 0.30
DEFAULT_MIN_CLUSTERS = 2
DEFAULT_UMAP_COMPONENTS = 5
DEFAULT_UMAP_NEIGHBORS = 15


def _fraction(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1: {text}")
    return value


def add_parsers(sub: argparse._SubParsersAction) -> None:
    positive = cli_common.positive_int
    parser = sub.add_parser("grid", help="reduce, cluster over the grid and choose a configuration")
    parser.add_argument("--max-noise", type=_fraction, default=DEFAULT_MAX_NOISE)
    parser.add_argument("--min-clusters", type=positive, default=DEFAULT_MIN_CLUSTERS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--umap-components", type=positive, default=DEFAULT_UMAP_COMPONENTS)
    parser.add_argument("--umap-neighbors", type=positive, default=DEFAULT_UMAP_NEIGHBORS)
    parser.add_argument("--grid-file", type=Path, default=None)
    parser = sub.add_parser("select", help="choose a grid configuration by hand")
    parser.add_argument("--config-id", required=True)


def _load_grid_config(path: Path | None) -> grid.GridConfig:
    if path is None:
        return grid.GridConfig()
    if not path.exists():
        raise FileNotFoundError(f"grid file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"grid file {path} is not valid JSON: {exc}") from None
    if not isinstance(raw, dict):
        raise ValueError(f"grid file {path} must hold a JSON object")
    return grid.GridConfig.from_mapping(raw)


def _write_review_for(
    *, root: Path, config_id: str, labels: np.ndarray, skills: Sequence[str], df: Sequence[int]
) -> None:
    rows = report.review_rows(labels, skills=skills, df=df)
    report.write_review(paths.taxonomy_dir(root), config_id, rows)


def _grid_context(
    *,
    args: argparse.Namespace,
    grid_config: grid.GridConfig,
    reduce_config: ReduceConfig,
    summary: Mapping[str, object],
    meta: Mapping[str, object],
) -> dict:
    return {
        "reduce": asdict(reduce_config),
        "grid": asdict(grid_config),
        "seed": args.seed,
        "max_noise": args.max_noise,
        "min_clusters": args.min_clusters,
        "selected_skills_version": summary["selected_skills_version"],
        "alias_table_version": summary["alias_table_version"],
        "embedding": meta["embedding"],
    }


def _check_grid_inputs(
    vectors: store.VectorSet, summary: Mapping[str, object], selected: Sequence[str]
) -> None:
    if vectors.meta.get("selected_skills_version") != summary["selected_skills_version"]:
        raise ValueError(
            "skill vectors were embedded for a different selected skill list "
            f"({vectors.meta.get('selected_skills_version')} vs "
            f"{summary['selected_skills_version']}); run taxonomy embed again"
        )
    if list(vectors.ids) != list(selected):
        raise ValueError(
            "skill vectors do not match the selected skills in rank order; "
            "run taxonomy embed again"
        )
    if len(vectors.ids) < MIN_ROWS:
        raise ValueError(
            f"only {len(vectors.ids)} skills selected; the grid needs at least {MIN_ROWS} "
            "(try a lower --min-df in taxonomy counts)"
        )


def _choose_and_save(
    args: argparse.Namespace, results: list[grid.ConfigResult], labels: np.ndarray, context: dict
) -> dict:
    root = args.artifacts_dir
    chosen = selection.choose(results, args.max_noise, args.min_clusters)
    if chosen is None:
        raise ValueError(
            f"no eligible configuration among {len(results)}; grid files were kept in "
            f"{paths.taxonomy_dir(root)}. Raise --max-noise or edit the grid with --grid-file"
        )
    n_eligible = len(selection.eligible(results, args.max_noise, args.min_clusters))
    record = selection.selection_record(
        chosen, source="auto", context=context, n_eligible=n_eligible
    )
    report.write_selection(paths.taxonomy_dir(root), record)
    skills, df = cli_common.selected_with_df(root)
    row = [r.config_id for r in results].index(chosen.config_id)
    _write_review_for(
        root=root, config_id=chosen.config_id, labels=labels[row], skills=skills, df=df
    )
    return record


def run_grid(args: argparse.Namespace, reduce_fn: ReduceFn) -> int:
    root = args.artifacts_dir
    grid_config = _load_grid_config(args.grid_file)
    summary = cli_common.read_counts_summary(root)
    vectors = cli_common.read_skill_vectors(root)
    _check_grid_inputs(vectors, summary, counts.read_selected(paths.skills_dir(root)))
    reduce_config = ReduceConfig(
        n_components=args.umap_components, n_neighbors=args.umap_neighbors, seed=args.seed
    )
    reduced = reduce_fn(vectors.vectors, reduce_config)
    results, labels = grid.run_grid(
        vectors.vectors,
        reduced,
        configs=grid.enumerate_configs(grid_config),
        seed=args.seed,
    )
    context = _grid_context(
        args=args,
        grid_config=grid_config,
        reduce_config=reduce_config,
        summary=summary,
        meta=vectors.meta,
    )
    out = paths.taxonomy_dir(root)
    report.write_grid(out, results, labels)
    report.write_context(out, context)
    print(json.dumps(_choose_and_save(args, results, labels, context)))
    return cli_common.EXIT_OK


def run_select(args: argparse.Namespace) -> int:
    root = args.artifacts_dir
    out = paths.taxonomy_dir(root)
    results, labels = report.read_grid(out)
    context = report.read_context(out)
    by_id = {r.config_id: i for i, r in enumerate(results)}
    if args.config_id not in by_id:
        known = out / report.GRID_RESULTS_NAME
        raise ValueError(f"unknown config id {args.config_id!r}; see {known}")
    chosen = results[by_id[args.config_id]]
    if chosen.skipped is not None:
        raise ValueError(f"config {chosen.config_id} was skipped in the grid: {chosen.skipped}")
    skills, df = cli_common.selected_with_df(root)
    if labels.shape[1] != len(skills):
        raise ValueError("the grid was run on a different skill list; run taxonomy grid again")
    record = selection.selection_record(chosen, source="manual", context=context, n_eligible=None)
    report.write_selection(out, record)
    _write_review_for(
        root=root,
        config_id=chosen.config_id,
        labels=labels[by_id[args.config_id]],
        skills=skills,
        df=df,
    )
    print(json.dumps(record))
    return cli_common.EXIT_OK

"""The plot subcommand: pictures of the chosen clustering, read from saved artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from getwork import fileio, paths
from getwork.taxonomy import cli_common, viz_data, viz_index
from getwork.taxonomy.build import NOISE_LABEL
from getwork.taxonomy.cli_common import EXIT_OK
from getwork.taxonomy.reduce import ReduceFn
from getwork.taxonomy.viz_index import IndexEntry

DEFAULT_SEED = 0
DEFAULT_MAX_LABEL_CHARS = 22
PLOTS_DIRNAME = "plots"
MAP_PNG, MAP_HTML = "cluster_map.png", "cluster_map.html"
SIZES_PNG, BALANCE_PNG = "cluster_sizes.png", "cluster_corpus_balance.png"
ONE_CORPUS_PNG, SKILLS_PNG = "cluster_one_corpus.png", "skill_document_frequency.png"
GRID_PNG, SUMMARY_CSV, INDEX_HTML = "grid_overview.png", "cluster_summary.csv", "index.html"


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("plot", help="draw the clusters and how skills relate to them")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument(
        "--max-label-chars", type=cli_common.positive_int, default=DEFAULT_MAX_LABEL_CHARS
    )


def default_out_dir(root: Path) -> Path:
    return paths.taxonomy_dir(root) / PLOTS_DIRNAME


def _index_entries(*, interactive: bool) -> list[IndexEntry]:
    return [
        IndexEntry(
            "Cluster map",
            "Each dot is a skill, colored by cluster (grey = noise). Positions come from a "
            "separate 2-D UMAP, so distances are only a rough guide to the clustering space.",
            image=MAP_PNG, interactive=MAP_HTML if interactive else None,
        ),
        IndexEntry("Cluster sizes", "Skills per cluster, largest first; noise on top.",
                   image=SIZES_PNG),
        IndexEntry(
            "Mentions by corpus (per cluster)",
            "One point per cluster. Points on an axis are mentioned in one corpus only.",
            image=BALANCE_PNG,
        ),
        IndexEntry(
            "Skills seen in one corpus only",
            "Skills per cluster that occur in both corpora, only resumes or only postings.",
            image=ONE_CORPUS_PNG,
        ),
        IndexEntry("Skill document frequencies",
                   "Every skill at (resume df, posting df), colored by cluster.", image=SKILLS_PNG),
        IndexEntry(
            "Grid overview",
            "All configurations: clusters vs noise, silhouette as color; the chosen one is ringed.",
            image=GRID_PNG,
        ),
    ]


def _summary_text(data: viz_data.PlotData) -> str:
    n_noise = int((data.labels == NOISE_LABEL).sum())
    return (
        f"Config {data.chosen_id}: {len(viz_data.cluster_ids(data.labels))} clusters, "
        f"{n_noise} of {len(data.skills)} skills are noise; "
        f"{viz_data.one_corpus_fraction(data):.1%} of skills appear in only one corpus."
    )


def _render(
    args: argparse.Namespace, data: viz_data.PlotData, coords: np.ndarray
) -> dict[str, object]:
    """Every output by file name (figures and the HTML page), built before anything is written."""
    from getwork.taxonomy import viz_html, viz_plots  # deferred: matplotlib and plotly are slow

    chars = args.max_label_chars
    rendered: dict[str, object] = {
        MAP_PNG: viz_plots.cluster_map_figure(data, coords, max_label_chars=chars),
        SIZES_PNG: viz_plots.cluster_sizes_figure(data),
        BALANCE_PNG: viz_plots.corpus_balance_figure(data),
        ONE_CORPUS_PNG: viz_plots.corpus_only_figure(data),
        SKILLS_PNG: viz_plots.skill_scatter_figure(data),
        GRID_PNG: viz_plots.grid_overview_figure(data),
    }
    if not args.no_interactive:
        interactive = viz_html.cluster_map_figure(data, coords, max_label_chars=chars)
        rendered[MAP_HTML] = viz_html.figure_html(interactive)
    return rendered


def _write_text(path: Path, text: str) -> None:
    fileio.atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def _write_outputs(out_dir: Path, rendered: dict[str, object]) -> None:
    from getwork.taxonomy import viz_plots  # already imported by _render

    for name, item in rendered.items():
        if isinstance(item, str):
            _write_text(out_dir / name, item)
        else:
            viz_plots.save_png(item, out_dir / name)


def run_plot(args: argparse.Namespace, reduce_fn: ReduceFn) -> int:
    root = args.artifacts_dir
    out_dir = args.out_dir if args.out_dir is not None else default_out_dir(root)
    data = viz_data.load_plot_data(root)
    coords = viz_data.project_2d(data.vectors, seed=args.seed, reduce_fn=reduce_fn)
    rendered = _render(args, data, coords)
    _write_outputs(out_dir, rendered)
    table = viz_data.cluster_table(data)
    fileio.write_csv_atomic(
        out_dir / SUMMARY_CSV, viz_data.TABLE_COLUMNS,
        table[list(viz_data.TABLE_COLUMNS)].values.tolist(),
    )
    index = viz_index.build_index(
        title="Skill taxonomy plots", summary=_summary_text(data),
        entries=_index_entries(interactive=not args.no_interactive),
    )
    _write_text(out_dir / INDEX_HTML, index)
    files = sorted([*rendered, SUMMARY_CSV, INDEX_HTML])
    print(json.dumps({"out_dir": str(out_dir), "files": files}))
    return EXIT_OK

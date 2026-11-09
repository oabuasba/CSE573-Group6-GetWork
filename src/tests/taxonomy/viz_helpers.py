"""A small synthetic artifact tree (30 skills, clusters 0/1/3 plus noise) for the plot tests."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from getwork import fileio, paths
from getwork.embed import store
from getwork.taxonomy import counts, embed_skills, grid, report, select

SKILLS = [f"skill{i:02d}" for i in range(30)]
# cluster ids are not contiguous (0, 1, 3) so id and position cannot be confused
# sizes 5, 8, 12 and 5 noise: size order (3, 1, 0) differs from id order (0, 1, 3)
LABELS = np.array([0] * 5 + [1] * 8 + [3] * 12 + [-1] * 5, dtype=np.int32)
RESUMES_ONLY = frozenset({0, 1, 2, 5, 6, 20})
EXTRA_SKILLS = (("extra a", 7, 3), ("extra b", 2, 9), ("extra c", 4, 0))
NAMES = {0: "Alpha tools", 1: "Beta / Gamma", 3: "A very long cluster name here"}
CHOSEN = "dbscan-02"
VERSION = counts.skills_version(SKILLS)


def df_resumes(i: int) -> int:
    return 10 + i


def df_postings(i: int) -> int:
    return 0 if i in RESUMES_ONLY else 1 + i % 7


def _result(
    config_id: str, *, n_clusters: int, noise: float, sil: float | None
) -> grid.ConfigResult:
    return grid.ConfigResult(
        config_id=config_id, method=config_id.split("-")[0], params={},
        n_clusters=n_clusters, noise_fraction=noise, silhouette=sil,
        largest_cluster_fraction=0.3, skipped=None,
    )


def _grid_results() -> list[grid.ConfigResult]:
    skipped = grid.ConfigResult(
        config_id="kmeans-01", method="kmeans", params={"k": 60}, n_clusters=0,
        noise_fraction=1.0, silhouette=None, largest_cluster_fraction=0.0,
        skipped=grid.SKIP_REASON,
    )
    return [
        _result("dbscan-01", n_clusters=7, noise=0.1, sil=0.2),
        _result(CHOSEN, n_clusters=3, noise=5 / 30, sil=0.4),
        skipped,
    ]


def _write_counts(root: Path, version: str) -> None:
    """Selected skills plus unselected extras, written in a shuffled row order."""
    selected = [
        counts.SkillCount(s, df_resumes(i) + df_postings(i), df_resumes(i), df_postings(i))
        for i, s in enumerate(SKILLS)
    ]
    extras = [counts.SkillCount(n, r + p, r, p) for n, r, p in EXTRA_SKILLS]
    everything = [*selected, *extras]
    order = np.random.default_rng(5).permutation(len(everything))
    shuffled = [everything[k] for k in order]
    summary = {"selected_skills_version": version, "alias_table_version": "sha256:a"}
    counts.write_counts(paths.skills_dir(root), shuffled, selected, summary)


def _write_grid(root: Path, context_version: str, chosen: int) -> None:
    out = paths.taxonomy_dir(root)
    results = _grid_results()
    matrix = np.stack([LABELS, LABELS, np.full(len(SKILLS), grid.SKIPPED_LABEL)])
    report.write_grid(out, results, matrix)
    context = {"selected_skills_version": context_version, "seed": 0}
    report.write_context(out, context)
    record = select.selection_record(results[chosen], source="auto", context=context, n_eligible=2)
    report.write_selection(out, record)


def write_names(root: Path, sizes: dict[int, int] | None = None) -> None:
    sizes = sizes or {c: int((LABELS == c).sum()) for c in NAMES}
    path = paths.taxonomy_dir(root) / "cluster_names.csv"
    fileio.write_csv_atomic(
        path, ("cluster_id", "name", "size"), [[c, n, sizes[c]] for c, n in NAMES.items()]
    )


def make_tree(
    root: Path, *, vectors_version: str = VERSION, vector_ids: list[str] | None = None,
    chosen: int = 1,
) -> Path:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(SKILLS), 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    meta = {
        "selected_skills_version": vectors_version, "kind": "skills",
        "embedding": {"backend": "fake", "model": "fake", "dim": 8},
    }
    store.write_vectors(
        paths.skills_dir(root), embed_skills.SKILL_STEM, vector_ids or SKILLS, vectors, meta
    )
    _write_counts(root, VERSION)
    _write_grid(root, VERSION, chosen)
    write_names(root)
    return root


def tree_listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# Hand-written asymmetric cluster centers (x != y) for the coordinates fixture.
CENTERS = {0: (1.0, 50.0), 1: (200.0, 7.0), 3: (-30.0, 90.0), -1: (5.0, 5.0)}


def make_coords() -> np.ndarray:
    """Members sit symmetrically around their center, so each median is exactly the center."""
    coords = np.zeros((len(SKILLS), 2))
    for label, (cx, cy) in CENTERS.items():
        rows = np.flatnonzero(LABELS == label)
        offsets = (np.arange(len(rows)) - (len(rows) - 1) / 2) * 0.1
        coords[rows, 0] = cx + offsets
        coords[rows, 1] = cy + offsets
    return coords

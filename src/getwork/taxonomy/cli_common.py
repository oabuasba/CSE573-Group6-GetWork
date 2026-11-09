"""Pieces shared by the taxonomy subcommands: exit codes, argument types and saved state."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from getwork import fileio, paths
from getwork.embed import store
from getwork.taxonomy import counts, embed_skills, report

EXIT_OK = 0
EXIT_FALLBACK = 1
EXIT_USAGE = 2
EXIT_FATAL = 3


class FatalError(Exception):
    """A fatal LLM or embedding failure; the message is already redacted."""


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {text}")
    return value


def read_counts_summary(root: Path) -> dict:
    path = paths.skills_dir(root) / counts.SUMMARY_FILE
    if not path.exists():
        raise FileNotFoundError(f"counts summary not found: {path} (run taxonomy counts)")
    return fileio.read_json(path)


def read_skill_vectors(root: Path) -> store.VectorSet:
    try:
        return store.read_vectors(paths.skills_dir(root), embed_skills.SKILL_STEM)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc} (run taxonomy embed)") from None


def selected_with_df(root: Path) -> tuple[list[str], list[int]]:
    skills_dir = paths.skills_dir(root)
    skills = counts.read_selected(skills_dir)
    rows = pq.read_table(skills_dir / counts.COUNTS_FILE, columns=["skill", "df_total"])
    df_by_skill = dict(zip(rows["skill"].to_pylist(), rows["df_total"].to_pylist()))
    return skills, [df_by_skill[skill] for skill in skills]


@dataclass(frozen=True)
class SelectionState:
    selection: dict
    labels: np.ndarray  # the chosen configuration's row, aligned with skills
    skills: list[str]
    df: list[int]


def load_selection_state(root: Path) -> SelectionState:
    """The saved selection with its labels row; refuses a selection from another grid run."""
    out = paths.taxonomy_dir(root)
    selection = report.read_selection(out)
    if selection.get("context") != report.read_context(out):
        raise ValueError(
            "selection.json does not belong to the current grid (its context differs from "
            "grid_context.json); run taxonomy grid or taxonomy select again"
        )
    current = read_counts_summary(root)["selected_skills_version"]
    if selection["context"].get("selected_skills_version") != current:
        raise ValueError(
            "selection.json was made for a different selected skill list than the current "
            "counts; run taxonomy grid again"
        )
    results, matrix = report.read_grid(out)
    ids = [r.config_id for r in results]
    config_id = selection["config_id"]
    if config_id not in ids:
        raise ValueError(f"selected config {config_id} is not in {out / report.GRID_RESULTS_NAME}")
    skills, df = selected_with_df(root)
    if matrix.shape[1] != len(skills):
        raise ValueError("the grid was run on a different skill list; run taxonomy grid again")
    return SelectionState(
        selection=selection, labels=matrix[ids.index(config_id)], skills=skills, df=df
    )

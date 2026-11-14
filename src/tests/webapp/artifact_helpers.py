"""Synthetic artifact tree for the ArtifactEngine tests: retrieval chain plus alias table and taxonomy."""

import json
import sys
from pathlib import Path

from getwork import fileio, paths
from getwork.alias.cli import TABLE_FILE
from getwork.alias.table import AliasTable, make_table, save_table
from getwork.docembed.cli_build import TAXONOMY_FILE
from getwork.taxonomy.model import Taxonomy, make_taxonomy, save_taxonomy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "retrieval"))
from retrieval_helpers import make_chain, make_chain_taxonomy  # noqa: E402

ALIASES = {"js": "JavaScript", "py": "Python"}
P10 = 3


def _use_valid_via(root: Path) -> None:
    """The retrieval helper labels clusters via "lookup"; the API only knows the CC-15 values."""
    for source in paths.SOURCES:
        path = paths.documents_dir(root, source) / f"{source}_documents.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            for item in row["skill_clusters"]:
                item["via"] = "taxonomy"
        fileio.write_jsonl_atomic(path, rows)


def save_alias_and_taxonomy(
    root: Path, *, entries: dict[str, str] = ALIASES, tax: Taxonomy | None = None
) -> tuple[AliasTable, Taxonomy]:
    """Save an alias table and a taxonomy that points at it (tax supplies clusters and noise)."""
    base = tax or make_chain_taxonomy()
    table = make_table(entries)
    save_table(paths.alias_dir(root) / TABLE_FILE, table, {})
    linked = make_taxonomy(base.clusters, base.noise, table.version, {})
    save_taxonomy(paths.taxonomy_dir(root) / TAXONOMY_FILE, linked)
    return table, linked


def build_artifacts(
    root: Path, n_resumes: int = 6, n_postings: int = 5, embed: bool = True
) -> tuple[AliasTable, Taxonomy]:
    make_chain(root=root, n_resumes=n_resumes, n_postings=n_postings, dim=16, embed=embed)
    _use_valid_via(root)
    for source in paths.SOURCES:
        summary = paths.relabeled_dir(root, source) / f"{source}_relabeled_summary.json"
        fileio.write_json_atomic(summary, {"p10_threshold": P10})
    return save_alias_and_taxonomy(root)

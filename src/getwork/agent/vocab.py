"""Skill vocabularies for the grounding checks: built from artifacts or from the demo data."""

from pathlib import Path

from getwork import paths
from getwork.agent.grounding import Vocabulary, make_vocabulary
from getwork.alias.cli import TABLE_FILE
from getwork.alias.table import AliasTable, load_table
from getwork.taxonomy.model import load_taxonomy
from getwork.webapp.demo import DEMO_ALIASES, DEMO_CLUSTERS

TAXONOMY_FILE = "taxonomy.json"


def _load_alias_table(path: Path) -> AliasTable:
    try:
        return load_table(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"alias table not found: {path} (run alias-table build)") from e


def load_vocabulary(root: Path) -> Vocabulary:
    """Skills: the keys and values of load_table(paths.alias_dir(root) / TABLE_FILE).entries and every member
    skill (tax.clusters[*].skills) and noise skill (tax.noise) of load_taxonomy(paths.taxonomy_dir(root) /
    "taxonomy.json"); clusters: tax.cluster_names(). FileNotFoundError naming the path (hint: run alias-table
    build / taxonomy build)."""
    table = _load_alias_table(paths.alias_dir(root) / TABLE_FILE)
    tax = load_taxonomy(paths.taxonomy_dir(root) / TAXONOMY_FILE)
    # Canonical spellings come first so they win over the lower-cased alias keys as display names.
    skills = [*table.entries.values(), *table.entries]
    skills += [skill for cluster in tax.clusters for skill in cluster.skills]
    skills += tax.noise
    return make_vocabulary(skills, tax.cluster_names())


def demo_vocabulary() -> Vocabulary:
    """Skills: DEMO_ALIASES values, then DEMO_CLUSTERS keys (first spelling wins); clusters: DEMO_CLUSTERS values."""
    return make_vocabulary([*DEMO_ALIASES.values(), *DEMO_CLUSTERS], DEMO_CLUSTERS.values())

from pathlib import Path

import pytest

from getwork import paths
from getwork.agent import vocab
from getwork.alias.cli import TABLE_FILE
from getwork.alias.table import make_table, save_table
from getwork.taxonomy.model import Cluster, make_taxonomy, save_taxonomy

pytestmark = pytest.mark.usefixtures("offline")


def write_alias(root: Path) -> str:
    table = make_table({"reactjs": "React", "k8s": "Kubernetes"})
    save_table(paths.alias_dir(root) / TABLE_FILE, table, {})
    return table.version


def write_taxonomy(root: Path, alias_version: str) -> None:
    clusters = [
        Cluster(id=0, name="Front-end", skills=("React", "Svelte")),
        Cluster(id=1, name="DevOps", skills=("Docker",)),
    ]
    tax = make_taxonomy(clusters, ["Communication"], alias_version, {})
    save_taxonomy(paths.taxonomy_dir(root) / "taxonomy.json", tax)


def test_load_vocabulary(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "art"
    write_taxonomy(root, write_alias(root))
    loaded = vocab.load_vocabulary(root)
    assert "reactjs" in loaded.skills
    assert loaded.skills["react"] == "React"
    assert loaded.skills["kubernetes"] == "Kubernetes" and "k8s" in loaded.skills
    assert {"svelte", "docker", "communication"} <= set(loaded.skills)
    assert loaded.clusters == frozenset({"front-end", "devops"})
    assert loaded.max_tokens == 1


def test_missing_files_have_hints(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "art"
    with pytest.raises(FileNotFoundError) as no_alias:
        vocab.load_vocabulary(root)
    assert str(paths.alias_dir(root) / TABLE_FILE) in str(no_alias.value)
    assert "alias-table build" in str(no_alias.value)
    write_alias(root)
    with pytest.raises(FileNotFoundError) as no_taxonomy:
        vocab.load_vocabulary(root)
    assert str(paths.taxonomy_dir(root) / "taxonomy.json") in str(no_taxonomy.value)
    assert "taxonomy build" in str(no_taxonomy.value)


def test_demo_vocabulary() -> None:
    demo = vocab.demo_vocabulary()
    assert demo.skills["react"] == "React" and demo.skills["vue"] == "Vue"
    assert demo.skills["tableau"] == "tableau"  # DEMO_CLUSTERS keys are lower-cased, so cluster-only skills are too
    assert {"front-end", "devops"} <= demo.clusters
    assert not demo.clusters & {"react", "vue"}

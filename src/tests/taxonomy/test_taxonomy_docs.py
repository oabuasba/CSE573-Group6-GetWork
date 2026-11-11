import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "getwork" / "taxonomy"
HEAVY = ("torch", "sentence_transformers", "openai")
REQUIRED_STRINGS = [
    "taxonomy counts",
    "taxonomy grid",
    "cluster_names.csv",
    "artifacts_trial",
    "ADR-6",
]


def test_pyproject_dependencies_and_script() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = config["project"]["dependencies"]
    names = {re.split(r"[<>=!~\[ ]", dep, maxsplit=1)[0] for dep in dependencies}
    assert {"scikit-learn", "scipy", "umap-learn"} <= names
    assert config["project"]["scripts"]["taxonomy"] == "getwork.taxonomy.cli:main"
    assert config["tool"]["setuptools"]["package-data"]["getwork.taxonomy"] == ["prompts/*.md"]


def test_runbook_mentions_commands_and_steps() -> None:
    text = (PACKAGE / "README.md").read_text(encoding="utf-8")
    for needle in REQUIRED_STRINGS:
        assert needle in text, needle
    headings = [line for line in text.splitlines() if line.startswith("#")]
    for step in range(1, 7):
        assert any(f"TU-{step}" in heading for heading in headings), step


def test_readme_documents_concept_names() -> None:
    text = (PACKAGE / "README.md").read_text(encoding="utf-8")
    for needle in ["--allow-member-names", "member_list", "name_v2", "NAME_METHOD", "offline runs only"]:
        assert needle in text, needle
    headings = [line for line in text.splitlines() if line.startswith("#")]
    for step in ("AV-4", "AV-5"):
        assert any(step in heading for heading in headings), step


def test_readme_documents_parallel_naming() -> None:
    text = (PACKAGE / "README.md").read_text(encoding="utf-8")
    for needle in ["--workers", "NAME_WORKERS", "stderr", "2 or more", "differ from `--workers 1`",
                   "re-asked", "id order"]:
        assert needle in text, needle


def _imported_modules(line: str) -> list[str]:
    plain = re.match(r"\s*import\s+(.+)", line)
    if plain:
        parts = [part.strip().split(" as ")[0] for part in plain.group(1).split(",")]
        return [part.split(".")[0] for part in parts]
    from_import = re.match(r"\s*from\s+([\w.]+)\s+import", line)
    return [from_import.group(1).split(".")[0]] if from_import else []


def test_import_scanner_sees_every_name_in_a_multi_import() -> None:
    assert _imported_modules("import os, torch") == ["os", "torch"]
    assert _imported_modules("    import numpy as np, openai.types") == ["numpy", "openai"]
    assert _imported_modules("from sentence_transformers import X") == ["sentence_transformers"]
    assert _imported_modules("x = 1  # import torch") == []


def test_package_has_no_heavy_imports() -> None:
    sources = sorted(PACKAGE.rglob("*.py"))
    assert len(sources) >= 10
    for path in sources:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            heavy = [name for name in _imported_modules(line) if name in HEAVY]
            assert not heavy, f"{path.name}:{number} imports {heavy}"

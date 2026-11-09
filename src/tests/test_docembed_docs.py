import ast
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "getwork" / "docembed"
HEAVY = ("torch", "sentence_transformers", "openai")
REQUIRED_STRINGS = [
    "doc-embed build", "doc-embed embed", "embed-preflight", "artifacts_trial", "--b2-order",
    "--level-bands",
]
STEPS = ["DU-1", "DU-2", "DU-3", "DU-4", "DU-5"]


def _imported_modules(source: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module.split(".")[0])
    return names


def test_import_scanner_sees_every_form() -> None:
    assert _imported_modules("import os, torch") == ["os", "torch"]
    assert _imported_modules("import numpy as np, openai.types") == ["numpy", "openai"]
    assert _imported_modules("from sentence_transformers import X") == ["sentence_transformers"]
    assert _imported_modules("def f():\n    import torch\n") == ["torch"]
    assert _imported_modules("x = 1  # import torch") == []


def test_pyproject_script() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["scripts"]["doc-embed"] == "getwork.docembed.cli:main"


def test_package_has_no_heavy_imports() -> None:
    modules = sorted(PACKAGE.glob("*.py"))
    assert {p.name for p in modules} >= {"cli.py", "documents.py", "vectors.py", "levels.py"}
    for path in modules:
        heavy = [n for n in _imported_modules(path.read_text(encoding="utf-8")) if n in HEAVY]
        assert not heavy, f"{path.name} imports {heavy}"


def test_runbook_mentions_commands_and_steps() -> None:
    text = (PACKAGE / "README.md").read_text(encoding="utf-8")
    for needle in REQUIRED_STRINGS:
        assert needle in text, needle
    headings = [line for line in text.splitlines() if line.startswith("#")]
    for step in STEPS:
        assert sum(step in heading for heading in headings) == 1, step

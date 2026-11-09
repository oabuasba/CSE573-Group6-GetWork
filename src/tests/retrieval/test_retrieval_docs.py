import ast
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "getwork" / "retrieval"
HEAVY = ("torch", "sentence_transformers", "openai")
REQUIRED_STRINGS = ["retrieve run", "retrieve show", "B1+", "--weights-file", "artifacts_trial"]
STEPS = ["RV-1", "RV-2", "RV-3", "RV-4"]


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


def test_pyproject_script_and_scipy() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["scripts"]["retrieve"] == "getwork.retrieval.cli:main"
    dependencies = config["project"]["dependencies"]
    assert any(dep.split("[")[0].split(">")[0].split("=")[0].strip() == "scipy"
               for dep in dependencies)


def test_package_has_no_heavy_imports() -> None:
    modules = sorted(PACKAGE.glob("*.py"))
    assert {p.name for p in modules} >= {
        "cli.py", "run.py", "query.py", "inputs.py", "topk.py", "b1plus.py",
    }
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

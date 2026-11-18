import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from getwork.webapp import pseudonym

pytestmark = pytest.mark.usefixtures("offline")

NAME_SHAPE = re.compile(r"^[A-Z][a-z]+ [A-Z][a-z]+$")
WORD_SHAPE = re.compile(r"^[A-Z][a-z]+$")
FORBIDDEN_CALLS = {"open", "hash", "eval", "exec", "__import__"}
FORBIDDEN_ATTR_BASES = {"os", "random", "time", "sys", "socket"}


def test_same_id_same_name() -> None:
    assert pseudonym.display_name("r-123") == pseudonym.display_name("r-123")


def test_stable_across_processes(tmp_path: Path) -> None:
    code = "from getwork.webapp import pseudonym; print(pseudonym.display_name('r-123'))"
    outputs = []
    for seed in ("1", "2"):
        run = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(run.stdout.strip())
    assert outputs[0] == outputs[1] == pseudonym.display_name("r-123")


def test_golden_names() -> None:
    assert pseudonym.display_name("r-1") == "Helen Chen"
    assert pseudonym.display_name("r-2") == "Natalie Long"
    assert pseudonym.display_name("r-30000") == "Larry Anderson"


def test_name_shape_and_spread() -> None:
    for names in (pseudonym.FIRST_NAMES, pseudonym.LAST_NAMES):
        assert len(names) >= 150
        assert len(set(names)) == len(names)
        assert all(WORD_SHAPE.match(name) for name in names)
        assert not any(name.startswith("Mc") for name in names)
        assert all(name[1:].islower() for name in names)
    generated = [pseudonym.display_name(f"r-{n}") for n in range(1, 2001)]
    assert all(name and NAME_SHAPE.match(name) for name in generated)
    assert len(set(generated)) >= 1800


@pytest.mark.parametrize("doc_id", [None, "", "   "])
def test_blank_id_gives_none(doc_id: str | None) -> None:
    assert pseudonym.display_name(doc_id) is None


def test_module_is_offline_and_pure() -> None:
    tree = ast.parse(Path(pseudonym.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    bad_calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                bad_calls.append(func.id)
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id in FORBIDDEN_ATTR_BASES:
                    bad_calls.append(f"{func.value.id}.{func.attr}")
    assert imported <= {"hashlib", "__future__"}
    assert bad_calls == []

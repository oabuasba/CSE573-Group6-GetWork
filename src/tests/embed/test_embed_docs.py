import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

ROOT = Path(__file__).resolve().parents[3]
VOYAGER_NAMES = ("VOYAGER_API_KEY", "VOYAGER_BASE_URL", "VOYAGER_MODEL")


def test_pyproject_extra_and_script() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert any(dep.startswith("numpy") for dep in project["dependencies"])
    extra = project["optional-dependencies"]["embed"]
    assert {"sentence-transformers", "torch"} <= set(extra)
    heavy = ("sentence-transformers", "torch")
    assert not any(dep.startswith(heavy) for dep in project["dependencies"])
    assert project["scripts"]["embed-preflight"] == "getwork.embed.cli:main"


def test_env_example_documents_embedding() -> None:
    lines = (ROOT / ".env.example").read_text().splitlines()
    assigned = [line.split("=")[0] for line in lines if line and not line.startswith("#")]
    assert assigned == list(VOYAGER_NAMES) or sorted(assigned) == sorted(VOYAGER_NAMES)
    for name in ("EMBED_BACKEND", "EMBED_MODEL", "VOYAGER_EMBED_MODEL"):
        assert any(line.startswith(f"# {name}=") for line in lines), name


def test_runbook_has_checklist() -> None:
    text = (ROOT / "src/getwork/embed/README.md").read_text()
    for step in range(1, 7):
        assert f"EU-{step}" in text and any(
            line.startswith("#") and f"EU-{step}" in line for line in text.splitlines()
        )
    for needle in ("embed-preflight", "--extra embed", "inferred", "check_same_embedding"):
        assert needle in text, needle

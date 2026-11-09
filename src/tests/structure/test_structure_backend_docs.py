"""Doc and config-file checks for the local backend (reads .env.example, .gitignore, README)."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

ROOT = Path(__file__).resolve().parents[3]
README = ROOT / "src" / "getwork" / "structure" / "README.md"
VOYAGER_NAMES = {"VOYAGER_API_KEY", "VOYAGER_BASE_URL", "VOYAGER_MODEL"}
LOCAL_NAMES = ("LLM_BACKEND", "LOCAL_LLM_BASE_URL", "LOCAL_LLM_MODEL", "LOCAL_LLM_API_KEY")
README_PHRASES = (
    "--backend local",
    "--extra-body",
    "--max-input-chars",
    "inferred",
    "ProvenanceMismatchError",
    "one GPU",
)


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_env_example_documents_local_backend() -> None:
    lines = _lines(ROOT / ".env.example")
    assigned = {
        line.partition("=")[0].strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert assigned == VOYAGER_NAMES
    comments = [line for line in lines if line.lstrip().startswith("#")]
    for name in LOCAL_NAMES:
        assert any(re.match(rf"#\s*{name}=", line) for line in comments), name


def test_gitignore_ignores_local_trial_dirs() -> None:
    lines = [line.strip() for line in _lines(ROOT / ".gitignore")]
    assert "data/structured_trial_*/" in lines
    assert "data/structured_trial/" in lines
    assert ".env" in lines
    forbidden = {"data/structured/", "data/structured", ".env.example"}
    assert not forbidden & set(lines)


def test_runbook_has_local_checklist() -> None:
    text = README.read_text(encoding="utf-8")
    headings = re.findall(r"^#+\s+(L-\d+)\b", text, flags=re.MULTILINE)
    assert headings == [f"L-{n}" for n in range(1, 9)]
    for phrase in README_PHRASES:
        assert phrase in text, phrase

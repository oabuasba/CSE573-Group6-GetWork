"""Doc and config-file checks for the structurer (reads only .env.example, .gitignore, README)."""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

ROOT = Path(__file__).resolve().parents[3]
REQUIRED_NAMES = {"VOYAGER_API_KEY", "VOYAGER_BASE_URL", "VOYAGER_MODEL"}
MAX_PLACEHOLDER_KEY_CHARS = 20
REQUIRED_STRINGS = [
    "--preflight",
    "--dry-run",
    "--limit 5",
    "--sample 100",
    "--out-dir data/structured_trial",
    "--prompt-version",
    "VOYAGER_API_KEY",
]


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_env_example_placeholders_only() -> None:
    assigned = {}
    for line in _lines(ROOT / ".env.example"):
        if line.strip() and not line.lstrip().startswith("#"):
            name, _, value = line.partition("=")
            assigned[name.strip()] = value.strip()
    assert set(assigned) == REQUIRED_NAMES
    assert len(assigned["VOYAGER_API_KEY"]) < MAX_PLACEHOLDER_KEY_CHARS


def test_gitignore_rules() -> None:
    lines = [line.strip() for line in _lines(ROOT / ".gitignore")]
    assert ".env" in lines
    assert "data/structured_trial/" in lines
    assert not {".env.example", ".env*", ".env.*"} & set(lines)
    assert not [x for x in lines if x.startswith("data/structured/") or x == "data/structured"]


def test_runbook_mentions_checklist_commands() -> None:
    text = (ROOT / "src/getwork/structure/README.md").read_text(encoding="utf-8")
    for required in REQUIRED_STRINGS:
        assert required in text, required
    for step in range(1, 9):
        assert re.search(rf"^#+\s+U-{step}\b", text, re.MULTILINE), f"U-{step}"

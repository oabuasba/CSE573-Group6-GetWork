import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

REPO = Path(__file__).resolve().parents[2]
RUNBOOK = REPO / "src" / "getwork" / "webapp" / "README.md"
REQUIRED_STRINGS = (
    "webapp serve --demo", "webapp check", "artifacts_trial", "--artifacts-dir artifacts",
    "npm --prefix src/web ci", "npm --prefix src/web run build", "EMBED_BACKEND",
    "no authentication", "WEBAPP_ARTIFACTS_DIR", "Chat about this suggestion", "sessionStorage",
    "Hide suggestion",
)


def test_runbook_headings_and_commands() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    headings = re.findall(r"^## (WA-\d)\b", text, re.M)
    assert headings == [f"WA-{n}" for n in range(1, 7)]
    for needle in REQUIRED_STRINGS:
        assert needle in text, needle


def test_readme_and_gitignore() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    row = next(line for line in readme.splitlines() if line.startswith("| Web app"))
    assert row.rstrip().endswith("| Built |") and "webapp serve --demo" in row
    ignored = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "node_modules/" in ignored and "src/web/dist/" in ignored


def test_candidate_names_limitation_documented() -> None:
    deviations = (REPO / "docs" / "deviations.md").read_text(encoding="utf-8")
    assert any(line.startswith("| P-22 |") for line in deviations.splitlines())
    assert re.search(r"^## P-22\b", deviations, re.M)
    for needle in ("placeholder", "Demo name", "Row count: 22."):
        assert needle in deviations, needle
    runbook = RUNBOOK.read_text(encoding="utf-8")
    assert "Demo name" in runbook and "placeholder" in runbook
    architecture = (REPO / "specs" / "architecture.html").read_text(encoding="utf-8")
    assert 'id="ADR-40"' in architecture and 'id="CC-21"' in architecture

"""Runbook check for the alias table (reads only the README)."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

README = Path(__file__).resolve().parents[2] / "getwork" / "alias" / "README.md"
REQUIRED_STRINGS = [
    "alias-table build",
    "alias-table apply",
    "alias-table propose",
    "artifacts_trial",
    "alias_curated.csv",
]
STEPS = ["AU-1", "AU-2", "AU-3", "AU-4"]


def test_runbook_mentions_commands_and_steps() -> None:
    text = README.read_text(encoding="utf-8")
    for needle in REQUIRED_STRINGS:
        assert needle in text, needle
    headings = [line for line in text.splitlines() if line.startswith("#")]
    for step in STEPS:
        assert any(step in heading for heading in headings), step


REBUILD_COMMANDS = [
    "taxonomy counts", "taxonomy grid", "taxonomy name", "taxonomy build", "relabel",
    "doc-embed build", "retrieve run",
]


def test_runbook_mentions_version_families_and_rebuild_order() -> None:
    text = README.read_text(encoding="utf-8")
    required = ["version_families.csv", "alias_version_report.json", "--families", *REBUILD_COMMANDS]
    for needle in required:
        assert needle in text, needle
    headings = [line for line in text.splitlines() if line.startswith("#")]
    for step in ["AV-1", "AV-2", "AV-3"]:
        assert sum(step in heading for heading in headings) == 1, step

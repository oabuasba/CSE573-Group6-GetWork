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

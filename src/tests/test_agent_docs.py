from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

REPO = Path(__file__).resolve().parents[2]
RUNBOOK = REPO / "src" / "getwork" / "agent" / "README.md"
REQUIRED_STRINGS = [
    "webapp serve --demo",
    "--explainer auto",
    "--explainer template",
    "--agent-model",
    "uv sync --extra agent",
    "FakeLLMClient",
    "failure_reason",
    "ungrounded_skills",
    "error share",
    "must differ from",
    "agent_v3",
    "outside these documents",
]


def test_runbook_headings_and_commands() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    for number in range(1, 7):
        assert any(h.startswith(f"## MA-{number} ") for h in headings), number
    for required in REQUIRED_STRINGS:
        assert required in text, required
    assert "sk-" not in text


def test_readme_status() -> None:
    lines = (REPO / "README.md").read_text(encoding="utf-8").splitlines()
    rows = [line for line in lines if line.startswith("|") and "explanation agent" in line]
    assert len(rows) == 1 and rows[0].rstrip().endswith("| Built |")
    text = "\n".join(lines)
    assert "uv run webapp serve --demo" in text
    assert "--explainer auto" not in text or "--explainer template" in text
    assert "--agent-model" in text

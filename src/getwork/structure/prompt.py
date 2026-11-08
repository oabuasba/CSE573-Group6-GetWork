"""System prompt loading and user-message builders for the structurer."""

from functools import cache
from pathlib import Path

PROMPT_VERSION: str = "extract_v1"
MAX_ERROR_CHARS: int = 300


@cache
def load_system_prompt(version: str = PROMPT_VERSION) -> str:
    """Read prompts/<version>.md as UTF-8; FileNotFoundError if it does not exist."""
    path = Path(__file__).parent / "prompts" / f"{version}.md"
    return path.read_text(encoding="utf-8")


def build_user_message(source: str, title: str | None, text: str) -> str:
    if source == "resumes":
        return "SOURCE: resume\n\nTEXT:\n" + text
    if source == "postings":
        return "SOURCE: job posting\nTITLE: " + (title or "") + "\n\nTEXT:\n" + text
    raise ValueError(f"unknown source: {source!r}")


def build_retry_message(user: str, error: str) -> str:
    return (
        user
        + "\n\nYour previous reply was rejected: "
        + error[:MAX_ERROR_CHARS]
        + "\nReturn only the corrected JSON object."
    )

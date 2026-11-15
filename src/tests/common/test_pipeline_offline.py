import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("offline")

TESTS_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = TESTS_DIR.parent
PKG_DIR = SRC_DIR / "getwork"
REQUIRED_LINE = 'pytestmark = pytest.mark.usefixtures("offline")'
PREFIXES = ("pipeline", "alias", "embed", "taxonomy", "relabel", "docembed", "retrieval", "evaluation")
HEAVY_IMPORT = re.compile(
    r"^[ \t]*(?:import|from)[ \t]+(openai|httpx2?|sentence_transformers|torch)\b", re.MULTILINE
)
ALLOWED = {
    "openai": {PKG_DIR / "llm" / "voyager.py", PKG_DIR / "embed" / "voyager_embed.py"},
    "sentence_transformers": {PKG_DIR / "embed" / "st_client.py"},
    "torch": {PKG_DIR / "embed" / "st_client.py"},
    "httpx": {TESTS_DIR / "llm" / "test_llm_voyager.py", TESTS_DIR / "embed" / "test_embed_voyager.py"},
    "httpx2": {TESTS_DIR / "llm" / "test_llm_voyager.py", TESTS_DIR / "embed" / "test_embed_voyager.py"},
}


def test_all_pipeline_tests_use_offline_fixture() -> None:
    files = [f for prefix in PREFIXES for f in TESTS_DIR.rglob(f"test_{prefix}_*.py")]
    offenders = [f.name for f in files if REQUIRED_LINE not in f.read_text().splitlines()]
    assert not offenders, f"missing offline pytestmark: {offenders}"


def test_heavy_libraries_imported_only_in_allowed_modules() -> None:
    candidates = [*PKG_DIR.rglob("*.py"), *TESTS_DIR.rglob("*.py")]
    unexpected = [
        f"{f.relative_to(SRC_DIR)} imports {lib}"
        for f in candidates
        if f != Path(__file__)
        for lib in HEAVY_IMPORT.findall(f.read_text())
        if f not in ALLOWED[lib]
    ]
    assert not unexpected, f"unexpected heavy imports: {sorted(unexpected)}"

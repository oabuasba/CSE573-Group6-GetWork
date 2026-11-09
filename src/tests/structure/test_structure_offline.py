import importlib.util
import os
import re
import socket
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.usefixtures("offline")

TESTS_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = TESTS_DIR.parent
REQUIRED_LINE = 'pytestmark = pytest.mark.usefixtures("offline")'
SDK_IMPORT = re.compile(r"^[ \t]*(?:import|from)[ \t]+(?:openai|httpx2?)\b", re.MULTILINE)


def _load_conftest() -> ModuleType:
    spec = importlib.util.spec_from_file_location("conftest_under_test", TESTS_DIR / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_fixture_blocks_sockets() -> None:
    with pytest.raises(RuntimeError, match="network disabled"):
        socket.create_connection(("127.0.0.1", 9))
    with pytest.raises(RuntimeError, match="network disabled"):
        socket.socket().connect(("127.0.0.1", 9))


def test_apply_offline_unsets_env_set_earlier_and_changes_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    names = (
        "VOYAGER_API_KEY",
        "VOYAGER_BASE_URL",
        "VOYAGER_MODEL",
        "LLM_BACKEND",
        "LOCAL_LLM_BASE_URL",
        "LOCAL_LLM_MODEL",
        "LOCAL_LLM_API_KEY",
        "EMBED_BACKEND",
        "EMBED_MODEL",
        "VOYAGER_EMBED_MODEL",
    )
    for name in names:
        monkeypatch.setenv(name, "x")
    _load_conftest().apply_offline(monkeypatch, tmp_path)
    assert all(name not in os.environ for name in names)
    assert Path.cwd() == tmp_path.resolve()
    assert not Path(".env").exists()


def test_all_llm_and_structure_tests_use_offline_fixture() -> None:
    files = [*TESTS_DIR.rglob("test_llm_*.py"), *TESTS_DIR.rglob("test_structure_*.py")]
    offenders = [f.name for f in files if REQUIRED_LINE not in f.read_text().splitlines()]
    assert not offenders, f"missing offline pytestmark: {offenders}"


def test_openai_imported_only_in_voyager_module_and_its_test() -> None:
    allowed = {
        SRC_DIR / "getwork" / "llm" / "voyager.py",
        SRC_DIR / "getwork" / "embed" / "voyager_embed.py",
        TESTS_DIR / "llm" / "test_llm_voyager.py",
        TESTS_DIR / "embed" / "test_embed_voyager.py",
    }
    candidates = [*(SRC_DIR / "getwork").rglob("*.py"), *TESTS_DIR.rglob("*.py")]
    matches = {
        f for f in candidates
        if f != Path(__file__) and SDK_IMPORT.search(f.read_text())
    }
    assert matches <= allowed, f"unexpected openai/httpx/httpx2 imports: {sorted(matches - allowed)}"

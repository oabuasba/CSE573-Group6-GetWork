from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

from getwork import paths
from getwork.docembed.documents import DocEmbedMismatchError
from getwork.embed.client import EmbeddingMismatchError
from getwork.evaluation import args as eval_args
from getwork.evaluation import cli
from getwork.llm.client import FatalLLMError
from getwork.llm.settings import ConfigError
from getwork.webapp.engine import EngineLoadError

pytestmark = pytest.mark.usefixtures("offline")

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "getwork" / "evaluation"
HEAVY = {"openai", "httpx", "httpx2", "sentence_transformers", "torch", "langgraph"}
STUB_NAME = "stub_cmd"


def _imported_modules(source: str) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module.split(".")[0])
    return names


class _Stub:
    """Registers a stub command whose run() raises whatever `raises` holds."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.raises: BaseException | None = None
        self.calls = 0
        module = ModuleType(f"getwork.evaluation.{STUB_NAME}")
        module.COMMANDS = ("stub",)  # type: ignore[attr-defined]
        module.add_parser = self._add_parser  # type: ignore[attr-defined]
        module.run = self._run  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(cli, "COMMAND_MODULES", (STUB_NAME,))

    @staticmethod
    def _add_parser(sub) -> None:  # type: ignore[no-untyped-def]
        parser = sub.add_parser("stub")
        eval_args.add_common_args(parser)
        parser.add_argument("--need", required=True)

    def _run(self, args, ctx) -> int:  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return eval_args.EXIT_OK


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Stub:
    monkeypatch.chdir(tmp_path)
    return _Stub(monkeypatch)


def test_script_and_package_data(stub: _Stub, capsys: pytest.CaptureFixture[str]) -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["scripts"]["evaluate"] == "getwork.evaluation.cli:main"
    assert config["tool"]["setuptools"]["package-data"]["getwork.evaluation"] == ["prompts/*.md"]
    assert cli.main(["--help"]) == 0
    assert "stub" in capsys.readouterr().out
    parsed = cli.build_parser().parse_args(["stub", "--need", "x"])
    assert (parsed.out_dir, parsed.artifacts_dir, parsed.processed_dir) == (
        Path("eval/run"), paths.DEFAULT_ARTIFACTS_DIR, paths.DEFAULT_PROCESSED_DIR,
    )
    assert cli.main(["stub", "--need", "x"]) == 0


def test_usage_errors_exit_2_and_create_nothing(
    stub: _Stub, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = sorted(tmp_path.rglob("*"))
    for argv in (
        ["nope"], ["stub"], ["stub", "--need", "x", "--out-dir", "data/x"],
        ["stub", "--need", "x", "--out-dir", "artifacts_trial/x"],
        ["stub", "--need", "x", "--out-dir", str(tmp_path / "art" / "sub"),
         "--artifacts-dir", str(tmp_path / "art")],
    ):
        assert cli.main(argv) == 2, argv
        assert "error:" in capsys.readouterr().err
    assert sorted(tmp_path.rglob("*")) == before
    assert stub.calls == 0


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ValueError("v"), 2), (FileNotFoundError("f"), 2), (KeyError("k"), 2),
        (ConfigError("c"), 2), (DocEmbedMismatchError("d"), 2),
        (EmbeddingMismatchError("e"), 2), (EngineLoadError("l"), 2),
        (FatalLLMError("bad key"), 3),
    ],
)
def test_error_mapping(
    stub: _Stub, error: Exception, code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    stub.raises = error
    assert cli.main(["stub", "--need", "x"]) == code
    assert "error:" in capsys.readouterr().err


def test_no_heavy_imports_in_package() -> None:
    files = sorted(PACKAGE.rglob("*.py"))
    assert files
    for path in files:
        heavy = HEAVY & set(_imported_modules(path.read_text(encoding="utf-8")))
        assert not heavy, f"{path.name} imports {sorted(heavy)}"

"""Command line entry point for the evaluation harness (the evaluate script)."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

from getwork import paths
from getwork.docembed.documents import DocEmbedMismatchError
from getwork.embed.client import EmbeddingMismatchError
from getwork.evaluation import layout
from getwork.evaluation.args import EXIT_FATAL, EXIT_USAGE, Context
from getwork.evaluation.judges import JudgeFatalError
from getwork.llm.client import FatalLLMError, LLMClient
from getwork.llm.settings import ConfigError
from getwork.webapp.engine import EngineLoadError

COMMAND_MODULES: tuple[str, ...] = (
    "cmd_sample", "cmd_pool", "cmd_rubric", "cmd_judge", "cmd_metrics",
    "cmd_agreement", "cmd_tune", "cmd_ari", "cmd_report",
)
FATAL_ERRORS: tuple[type[BaseException], ...] = (FatalLLMError, JudgeFatalError)
USAGE_ERRORS: tuple[type[BaseException], ...] = (
    ValueError, FileNotFoundError, KeyError, ConfigError, DocEmbedMismatchError,
    EmbeddingMismatchError, EngineLoadError,
)
PROTECTED_DEFAULTS: tuple[Path, ...] = (
    paths.DEFAULT_STRUCTURED_DIR, paths.DEFAULT_PROCESSED_DIR, paths.DEFAULT_ARTIFACTS_DIR,
)


def _command_modules() -> dict[str, ModuleType]:
    by_command: dict[str, ModuleType] = {}
    for name in COMMAND_MODULES:
        module = importlib.import_module(f"getwork.evaluation.{name}")
        by_command.update({command: module for command in module.COMMANDS})
    return by_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate", description="Evaluate B1+, B2 and Full: sample, pool, judge, score."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for module in dict.fromkeys(_command_modules().values()):
        module.add_parser(sub)
    return parser


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_USAGE


def _parse(argv: list[str] | None) -> argparse.Namespace | int:
    try:
        return build_parser().parse_args(argv)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else EXIT_USAGE


def _protected_paths(args: argparse.Namespace) -> list[Path]:
    return [Path("data"), args.artifacts_dir, args.processed_dir, *PROTECTED_DEFAULTS]


def main(
    argv: list[str] | None = None,
    *,
    llm_client: LLMClient | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Exit codes: 0 ok, 1 the judge run finished with failed pairs, 2 usage, config or
    input error, 3 fatal LLM error."""
    args = _parse(argv)
    if isinstance(args, int):
        return args
    ctx = Context(llm_client=llm_client, environ=os.environ if environ is None else environ)
    try:
        layout.check_out_dir(args.out_dir, _protected_paths(args))
        return _command_modules()[args.command].run(args, ctx)
    except FATAL_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FATAL
    except USAGE_ERRORS as exc:
        return _error(str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

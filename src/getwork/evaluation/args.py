"""Exit codes, the command context and the options every evaluate command shares."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from getwork import paths
from getwork.evaluation import layout
from getwork.llm.client import LLMClient

EXIT_OK, EXIT_FAILED_ITEMS, EXIT_USAGE, EXIT_FATAL = 0, 1, 2, 3


@dataclass(frozen=True)
class Context:
    llm_client: LLMClient | None
    environ: Mapping[str, str]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, default=layout.DEFAULT_OUT_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=paths.DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=paths.DEFAULT_PROCESSED_DIR)

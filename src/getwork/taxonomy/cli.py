"""The taxonomy command: counts, embed, grid, select, name and build."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from getwork import paths
from getwork.alias import apply
from getwork.embed import factory
from getwork.embed.client import EmbeddingClient, EmbeddingError
from getwork.llm.client import LLMClient, LLMError, redact_secrets
from getwork.llm.settings import ConfigError
from getwork.taxonomy import cli_common, cli_grid, cli_name, counts, embed_skills
from getwork.taxonomy.cli_common import EXIT_FATAL, EXIT_OK, EXIT_USAGE, FatalError
from getwork.taxonomy.reduce import ReduceFn, umap_reduce

DEFAULT_TOP_N = 1000
DEFAULT_MIN_DF = 2
DEFAULT_CHUNK_SIZE = 256
EMBED_CACHE_NAME = "_embed_cache"
SECRET_ENV_VAR = "VOYAGER_API_KEY"


def _add_counts_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("counts", help="count skills and keep the top N")
    parser.add_argument("--top-n", type=cli_common.positive_int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-df", type=cli_common.positive_int, default=DEFAULT_MIN_DF)


def _add_embed_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("embed", help="embed the selected skills")
    factory.add_embed_args(parser)
    parser.add_argument("--chunk-size", type=cli_common.positive_int, default=DEFAULT_CHUNK_SIZE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="taxonomy", description="Build the skill taxonomy.")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_counts_parser(sub)
    _add_embed_parser(sub)
    cli_grid.add_parsers(sub)
    cli_name.add_parsers(sub)
    for p in sub.choices.values():
        p.add_argument("--artifacts-dir", type=Path, default=paths.DEFAULT_ARTIFACTS_DIR)
    return parser


def _run_counts(args: argparse.Namespace) -> int:
    root = args.artifacts_dir
    records = {source: apply.read_canonical(root, source) for source in paths.SOURCES}
    version = apply.single_alias_version(records)
    all_counts = counts.count_skills(records)
    selected = counts.select_top(all_counts, args.top_n, args.min_df)
    summary = counts.counts_summary(
        all_counts,
        selected,
        n_docs={source: len(rows) for source, rows in records.items()},
        alias_table_version=version,
        top_n=args.top_n,
        min_df=args.min_df,
    )
    counts.write_counts(paths.skills_dir(root), all_counts, selected, summary)
    print(json.dumps(summary))
    return EXIT_OK


def _progress(done: int, total: int) -> None:
    print(f"embedded chunk {done}/{total}", file=sys.stderr)


def _make_bundle(
    args: argparse.Namespace, embed_client: EmbeddingClient | None, environ: Mapping[str, str]
) -> factory.EmbeddingBundle:
    """Config errors (ConfigError) propagate as exit 2; construction failures are fatal (3)."""
    try:
        return factory.make_embedding_client(args=args, environ=environ, client=embed_client)
    except (EmbeddingError, LLMError) as exc:
        known_secrets = [environ.get(SECRET_ENV_VAR, "")]
        raise FatalError(redact_secrets(str(exc), known_secrets)) from exc


def _run_embed(
    args: argparse.Namespace, embed_client: EmbeddingClient | None, environ: Mapping[str, str]
) -> int:
    bundle = _make_bundle(args, embed_client, environ)
    root = args.artifacts_dir
    skills_dir = paths.skills_dir(root)
    selected = counts.read_selected(skills_dir)
    summary = cli_common.read_counts_summary(root)
    meta_extra = {
        "selected_skills_version": summary["selected_skills_version"],
        "alias_table_version": summary["alias_table_version"],
    }
    try:
        embed_skills.embed_skills(
            bundle.client,
            selected,
            skills_dir,
            meta_extra=meta_extra,
            cache_dir=skills_dir / EMBED_CACHE_NAME,
            chunk_size=args.chunk_size,
            progress=_progress,
        )
    except (EmbeddingError, LLMError) as exc:
        raise FatalError(redact_secrets(str(exc), bundle.secrets)) from exc
    result = {"n_skills": len(selected), "embedding": bundle.info.as_dict(), **meta_extra}
    print(json.dumps(result))
    return EXIT_OK


def _error(message: str, environ: Mapping[str, str]) -> int:
    text = redact_secrets(message, [environ.get(SECRET_ENV_VAR, "")])
    print(f"error: {text}", file=sys.stderr)
    return EXIT_USAGE


def _parse(argv: list[str] | None) -> argparse.Namespace | int:
    try:
        return build_parser().parse_args(argv)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else EXIT_USAGE


def main(
    argv: list[str] | None = None,
    *,
    embed_client: EmbeddingClient | None = None,
    llm_client: LLMClient | None = None,
    environ: Mapping[str, str] | None = None,
    reduce_fn: ReduceFn = umap_reduce,
) -> int:
    """Exit codes: 0 ok, 1 name finished with failed clusters (heuristic names filled in),
    2 usage, config or input error, 3 fatal LLM or embedding error."""
    args = _parse(argv)
    if isinstance(args, int):
        return args
    environ = os.environ if environ is None else environ
    try:
        if args.command == "counts":
            return _run_counts(args)
        if args.command == "embed":
            return _run_embed(args, embed_client, environ)
        if args.command == "grid":
            return cli_grid.run_grid(args, reduce_fn)
        if args.command == "select":
            return cli_grid.run_select(args)
        if args.command == "name":
            return cli_name.run_name(args, llm_client, environ)
        return cli_name.run_build(args)
    except FatalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FATAL
    except KeyError as exc:
        return _error(f"an input file is missing the key {exc}", environ)
    except (ConfigError, OSError, ValueError) as exc:
        return _error(str(exc), environ)


if __name__ == "__main__":
    raise SystemExit(main())

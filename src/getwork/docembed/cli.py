"""The doc-embed command: build the documents tables, then embed them into vector sets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from getwork import paths
from getwork.docembed import cli_build, documents, vectors
from getwork.embed import factory
from getwork.embed.client import EmbeddingClient, EmbeddingError
from getwork.llm.client import LLMError, redact_secrets
from getwork.llm.settings import ConfigError

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_EMBED_FAILED = 3
SOURCE_CHOICES = ("resumes", "postings", "both")
DEFAULT_CHUNK_SIZE = 256
EMBED_CACHE_NAME = "_embed_cache"
SECRET_ENV_VAR = "VOYAGER_API_KEY"
PROGRESS_STEP_PERCENT = 10


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc-embed", description="Build and embed the document texts (pipeline step 6)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    cli_build.add_build_parser(sub, source_choices=SOURCE_CHOICES)
    embed = sub.add_parser("embed", help="embed the documents tables (user-run on the GPU)")
    embed.add_argument("--source", choices=SOURCE_CHOICES, default="both")
    embed.add_argument("--chunk-size", type=_positive_int, default=DEFAULT_CHUNK_SIZE)
    embed.add_argument("--no-cache", action="store_true")
    factory.add_embed_args(embed)
    for p in sub.choices.values():
        p.add_argument("--artifacts-dir", type=Path, default=paths.DEFAULT_ARTIFACTS_DIR)
    return parser


class _Progress:
    """Prints one stderr line per system each time another ten percent of chunks is done."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._last_step: dict[str, int] = {}

    def __call__(self, system: str, done: int, total: int) -> None:
        step = done * 100 // total // PROGRESS_STEP_PERCENT
        if step <= self._last_step.get(system, 0):
            return
        self._last_step[system] = step
        print(
            f"{self._source} {system}: {step * PROGRESS_STEP_PERCENT}% ({done}/{total} chunks)",
            file=sys.stderr,
        )


def _read_inputs(*, root: Path, sources: tuple[str, ...]) -> dict[str, tuple[dict, dict]]:
    """Documents and summary per source; refuses summaries that do not describe their table."""
    inputs = {}
    for source in sources:
        docs = documents.read_documents(root, source)
        summary = documents.read_documents_summary(root, source)
        missing = [k for k in ("n_documents", *documents.COMPARE_KEYS) if k not in summary]
        if missing:
            raise ValueError(f"{source}: summary lacks {missing}; {documents.BUILD_HINT}")
        if summary["n_documents"] != len(docs):
            raise ValueError(
                f"{source}: summary counts {summary['n_documents']} documents but the table "
                f"has {len(docs)}; {documents.BUILD_HINT}"
            )
        inputs[source] = (docs, summary)
    if len(inputs) == 2:
        documents.check_comparable(
            a=inputs["resumes"][1], b=inputs["postings"][1]
        )
    return inputs


class _EmbedFailed(Exception):
    """An embedding failure whose message is already redacted."""


def _make_bundle(
    *, args: argparse.Namespace, client: EmbeddingClient | None, environ: Mapping[str, str]
) -> factory.EmbeddingBundle:
    try:
        return factory.make_embedding_client(args=args, environ=environ, client=client)
    except (EmbeddingError, LLMError) as exc:
        secrets = [environ.get(SECRET_ENV_VAR, "")]
        raise _EmbedFailed(redact_secrets(text=str(exc), secrets=secrets)) from exc


def _run_embed(
    *, args: argparse.Namespace, client: EmbeddingClient | None, environ: Mapping[str, str]
) -> int:
    root = args.artifacts_dir
    sources = paths.SOURCES if args.source == "both" else (args.source,)
    inputs = _read_inputs(root=root, sources=sources)
    bundle = _make_bundle(args=args, client=client, environ=environ)
    for source, (docs, summary) in inputs.items():
        out_dir = paths.documents_dir(root, source)
        try:
            result = vectors.embed_documents(
                client=bundle.client,
                docs=docs,
                summary=summary,
                source=source,
                out_dir=out_dir,
                cache_dir=None if args.no_cache else out_dir / EMBED_CACHE_NAME,
                chunk_size=args.chunk_size,
                progress=_Progress(source),
            )
        except (EmbeddingError, LLMError) as exc:
            raise _EmbedFailed(redact_secrets(text=str(exc), secrets=bundle.secrets)) from exc
        report = {"source": source, "embedding": bundle.info.as_dict(), "systems": result}
        print(json.dumps(report))
    return EXIT_OK


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_USAGE


def _parse(argv: list[str] | None) -> argparse.Namespace | int:
    try:
        return build_parser().parse_args(argv)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else EXIT_USAGE


def main(
    argv: list[str] | None = None,
    *,
    client: EmbeddingClient | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Exit codes: 0 ok, 2 usage, config or input error, 3 embedding error."""
    args = _parse(argv)
    if isinstance(args, int):
        return args
    try:
        if args.command == "build":
            return cli_build.run_build(args)
        return _run_embed(
            args=args, client=client, environ=os.environ if environ is None else environ
        )
    except _EmbedFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_EMBED_FAILED
    except (ConfigError, documents.DocEmbedMismatchError, OSError, ValueError) as exc:
        return _error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

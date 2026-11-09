"""Command line entry point for retrieval (run, show)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from getwork import paths
from getwork.docembed.documents import DocEmbedMismatchError
from getwork.embed.client import EmbeddingMismatchError
from getwork.retrieval import b1plus, inputs, query, run

EXIT_OK = 0
EXIT_USAGE = 2
ALL_SYSTEMS = "all"
SHOW_DEFAULT_SYSTEM = "Full"
SHOW_DEFAULT_K = 5
MAX_SHARED_SKILLS = 8
_ERRORS = (
    FileNotFoundError, ValueError, KeyError, DocEmbedMismatchError, EmbeddingMismatchError,
)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {value}")
    return number


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifacts-dir", type=Path, default=paths.DEFAULT_ARTIFACTS_DIR)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--weights", help="B1+ weights skills,title,level, e.g. 0.7,0.15,0.15")
    group.add_argument("--weights-file", type=Path, help="JSON file with skills, title, level")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retrieve", description="Rank postings for resumes and resumes for postings."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="rank both directions for the chosen systems")
    _add_common_args(run_parser)
    run_parser.add_argument("--systems", default=ALL_SYSTEMS, help="all or a list like B1+,Full")
    run_parser.add_argument("--k", type=_positive_int, default=run.DEFAULT_K)
    run_parser.add_argument("--block-size", type=_positive_int, default=512)
    show = sub.add_parser("show", help="print one query's matches")
    _add_common_args(show)
    show.add_argument("--direction", required=True, choices=inputs.DIRECTIONS)
    show.add_argument("--id", required=True, dest="query_id")
    show.add_argument("--system", choices=inputs.SYSTEM_IDS, default=SHOW_DEFAULT_SYSTEM)
    show.add_argument("--k", type=_positive_int, default=SHOW_DEFAULT_K)
    return parser


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_USAGE


def _parse_systems(text: str) -> list[str]:
    if text == ALL_SYSTEMS:
        return list(inputs.SYSTEM_IDS)
    systems = [part.strip() for part in text.split(",")]
    unknown = [s for s in systems if s not in inputs.SYSTEM_IDS]
    if unknown:
        raise ValueError(f"unknown system {unknown}; choose from {list(inputs.SYSTEM_IDS)}")
    return systems


def _resolve_weights(args: argparse.Namespace) -> b1plus.B1PlusWeights:
    if args.weights is not None:
        return b1plus.parse_weights(args.weights)
    if args.weights_file is not None:
        return b1plus.load_weights(args.weights_file)
    return b1plus.DEFAULT_WEIGHTS


def _run_run(args: argparse.Namespace) -> int:
    systems = _parse_systems(args.systems)
    weights = _resolve_weights(args)
    data = inputs.load_inputs(root=args.artifacts_dir, systems=systems)
    summary = run.run_retrieval(
        inputs=data, root=args.artifacts_dir, systems=systems, k=args.k, weights=weights,
        block_size=args.block_size,
    )
    print(json.dumps(summary))
    return EXIT_OK


def _describe(*, doc_id: str, doc: dict) -> str:
    return f"{doc_id}  [{doc['level']}]  {doc['title']}"


def _shared_skills(*, query_doc: dict, match_doc: dict) -> list[str]:
    wanted = b1plus.skill_keys(list(query_doc["skills"]))
    shared = [s for s in match_doc["skills"] if b1plus.skill_keys([s]) & wanted]
    return shared[:MAX_SHARED_SKILLS]


def _run_show(args: argparse.Namespace) -> int:
    weights = _resolve_weights(args)
    data = inputs.load_inputs(root=args.artifacts_dir, systems=[args.system])
    matches = query.query_matches(
        inputs=data, system=args.system, direction=args.direction, query_id=args.query_id,
        k=args.k, weights=weights,
    )
    query_docs, match_docs = (
        (data.resumes, data.postings)
        if args.direction == "resume_to_posting"
        else (data.postings, data.resumes)
    )
    query_doc = query_docs[args.query_id]
    print(f"query {_describe(doc_id=args.query_id, doc=query_doc)}")
    print(f"  skills: {', '.join(query_doc['skills'])}")
    for rank, (doc_id, score) in enumerate(matches, start=1):
        shared = ", ".join(_shared_skills(query_doc=query_doc, match_doc=match_docs[doc_id]))
        line = _describe(doc_id=doc_id, doc=match_docs[doc_id])
        print(f"{rank:>3}  {score:.4f}  {line}  | shared: {shared}")
    return EXIT_OK


def _parse(argv: list[str] | None) -> argparse.Namespace | int:
    try:
        return build_parser().parse_args(argv)
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else EXIT_USAGE


def main(argv: list[str] | None = None) -> int:
    """Exit codes: 0 ok, 2 usage, config or input error."""
    args = _parse(argv)
    if isinstance(args, int):
        return args
    try:
        return _run_run(args) if args.command == "run" else _run_show(args)
    except _ERRORS as exc:
        return _error(str(exc.args[0]) if isinstance(exc, KeyError) else str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

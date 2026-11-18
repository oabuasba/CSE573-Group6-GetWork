"""evaluate judge: run the two judges over a stage's pairs and rebuild the label table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from getwork import fileio, paths
from getwork.evaluation import args as eval_args
from getwork.evaluation import (
    cmd_rubric,
    defaults,
    judge_run,
    judges,
    layout,
    llm_judge,
    loading,
)
from getwork.evaluation.args import (
    EXIT_FAILED_ITEMS,
    EXIT_FATAL,
    EXIT_OK,
    EXIT_USAGE,
    Context,
)
from getwork.evaluation.judges import JudgeInfo, PairText, Rubric
from getwork.llm import factory
from getwork.llm.client import LLMError, describe_exception
from getwork.llm.settings import BACKENDS, DEFAULT_BACKEND
from getwork.webapp import artifacts
from getwork.webapp.engine import EngineLoadError

COMMANDS = ("judge",)
DEFAULT_WORKERS = 4
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_MAX_CHARS = 20000
DEFAULT_MAX_TOKENS = 512


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return number


def _non_negative_int(value: str) -> int:
    return 0 if value.strip() == "0" else _positive_int(value)


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("judge", help="run the two judges over the pairs of a stage")
    eval_args.add_common_args(parser)
    parser.add_argument("--stage", choices=layout.STAGES, default=None)
    parser.add_argument("--judge-models", default=None, help="two model names A,B")
    parser.add_argument("--judge-families", default=None, help="two family names FA,FB")
    parser.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS)
    parser.add_argument("--max-chars", type=_positive_int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--max-tokens", type=_positive_int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--reasoning-effort", choices=defaults.REASONING_CHOICES, default="auto")
    parser.add_argument("--also-used-model", action="append", default=[],
                        help="another model the pipeline uses; judges must differ from it")
    parser.add_argument("--rubric-file", type=Path, default=None)
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--offset", type=_non_negative_int, default=0,
                        help="skip the first N pairs by sorted pair id (a held-out slice with "
                             "--limit); default 0")
    parser.add_argument("--progress-every", type=_non_negative_int, default=DEFAULT_PROGRESS_EVERY,
                        help="print a progress line to stderr every N judged pairs; 0 disables")
    parser.add_argument("--dry-run", action="store_true", help="count pairs and calls only")
    parser.add_argument("--preflight", action="store_true",
                        help="check both judge models against the provider's model list")
    parser.add_argument("--backend", choices=BACKENDS, default=None)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--extra-body", default=None, help="JSON object merged into every request")


def _two_names(option: str, raw: str | None, fallback: tuple[str, str]) -> tuple[str, str]:
    if raw is None:
        return fallback
    names = [name.strip() for name in raw.split(",")]
    if len(names) != 2 or not all(names):
        raise ValueError(f"{option} needs exactly two non-empty comma-separated names")
    return names[0], names[1]


def _judge_ids_and_names(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """(id, model, family) per judge; the two options come together or not at all."""
    if (args.judge_models is None) != (args.judge_families is None):
        raise ValueError("--judge-models and --judge-families must be given together")
    models = _two_names("--judge-models", args.judge_models, defaults.DEFAULT_JUDGE_MODELS)
    families = _two_names("--judge-families", args.judge_families, defaults.DEFAULT_JUDGE_FAMILIES)
    return list(zip(defaults.DEFAULT_JUDGE_IDS, models, families))


def _check_frozen(out_dir: Path, rubric: Rubric) -> None:
    path = out_dir / layout.RUBRIC_FROZEN_FILE
    if not path.exists():
        raise ValueError("the test stage needs a frozen rubric (run evaluate freeze-rubric)")
    if fileio.read_json(path).get("rubric_hash") != rubric.hash:
        raise ValueError("the rubric changed since it was frozen; the test stage needs the "
                         "frozen rubric")


def _check_manifest(out_dir: Path, infos: list[JudgeInfo]) -> bool:
    """True when the judges are already recorded; ValueError when they differ."""
    recorded = layout.read_manifest(out_dir).get("judges")
    if recorded is None:
        return False
    pick = lambda items: [(i["model"], i["family"]) for i in items]  # noqa: E731
    if pick(recorded) != [(i.model, i.family) for i in infos]:
        raise ValueError(f"{layout.MANIFEST_FILE} records other judges than the arguments; "
                         f"use the recorded models and families")
    return True


def _read_pair_texts(args: argparse.Namespace, stage: str) -> list[PairText]:
    path = layout.pairs_path(args.out_dir, stage)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (run evaluate pool --stage {stage})")
    table = pq.read_table(path, columns=["pair_id", "resume_id", "posting_id"]).to_pylist()
    table.sort(key=lambda row: row["pair_id"])
    end = None if args.limit is None else args.offset + args.limit
    table = table[args.offset : end]
    texts = {
        source: loading.read_texts(
            args.processed_dir, source, sorted({row[f"{source[:-1]}_id"] for row in table})
        )
        for source in paths.SOURCES
    }
    return [PairText(row["pair_id"], texts["resumes"][row["resume_id"]],
                     texts["postings"][row["posting_id"]]) for row in table]


def _already_done(out_dir: Path, rubric: Rubric, pairs: list[PairText]) -> int:
    ok_sets = []
    for judge_id in defaults.DEFAULT_JUDGE_IDS:
        rows = judge_run.read_judge_rows(layout.judge_path(out_dir, judge_id), rubric.hash)
        ok_sets.append({k for k, row in rows.items() if row["status"] == judge_run.OK_STATUS})
    return sum(1 for pair in pairs if all(pair.pair_id in done for done in ok_sets))


def _judge_all(
    args: argparse.Namespace, judges_: list[llm_judge.LLMJudge], pairs: list[PairText],
    rubric: Rubric,
) -> tuple[dict, bool, bool]:
    """(per-judge summary, any failed, any fatal); stops after the first fatal judge."""
    summary: dict = {}
    failed = False
    for judge in judges_:
        run = judge_run.run_judge(
            judge, pairs, path=layout.judge_path(args.out_dir, judge.info.id), rubric=rubric,
            workers=args.workers, progress_every=args.progress_every,
        )
        summary[judge.info.id] = {
            "model": judge.info.model, "family": judge.info.family, "pairs": run.n_pairs,
            "skipped": run.n_skipped, "ok": run.n_ok, "failed": run.n_failed,
            "truncated": run.n_truncated, "schema_fallback": judge.schema_fallback,
        }
        failed = failed or run.n_failed > 0
        if run.fatal:
            print(f"error: judge {judge.info.id} stopped: {run.fatal}", file=sys.stderr)
            return summary, failed, True
    return summary, failed, False


def _build_judges(
    args: argparse.Namespace, ctx: Context, names: list[tuple[str, str, str]], rubric: Rubric,
    extra_body: dict | None,
) -> list[llm_judge.LLMJudge]:
    built = []
    for judge_id, model, family in names:
        bundle = factory.make_client(
            backend=args.backend, model=model, env_file=Path(args.env_file),
            environ=ctx.environ, client=ctx.llm_client,
        )
        info = JudgeInfo(id=judge_id, model=model, family=family, backend=bundle.backend)
        built.append(llm_judge.LLMJudge(
            bundle.client, info, rubric, max_chars=args.max_chars, max_tokens=args.max_tokens,
            reasoning_effort=defaults.resolve_reasoning(model, args.reasoning_effort),
            use_schema=True, extra_body=extra_body,
        ))
    return built


def _checked_models(args: argparse.Namespace) -> list[str] | None:
    """Always-checked plus pipeline models; None when the artifacts root does not resolve."""
    try:
        root = artifacts.resolve_artifacts_root(args.artifacts_dir)
    except EngineLoadError as exc:
        print(f"note: canonical models not checked: {exc}", file=sys.stderr)
        return None
    return sorted({*defaults.DEFAULT_ALSO_USED_MODELS,
                   *loading.pipeline_models(root, args.also_used_model)})


def _preflight(args: argparse.Namespace, ctx: Context, names: list[tuple[str, str, str]]) -> int:
    bundles = [
        factory.make_client(
            backend=args.backend, model=model, env_file=Path(args.env_file),
            environ=ctx.environ, client=ctx.llm_client,
        )
        for _, model, _ in names
    ]
    infos = [JudgeInfo(id=i, model=m, family=f, backend=bundles[0].backend) for i, m, f in names]
    checked = _checked_models(args)
    judges.check_independence(infos, checked if checked is not None
                              else sorted({*defaults.DEFAULT_ALSO_USED_MODELS,
                                           *args.also_used_model}))
    missing = []
    for (judge_id, model, family), bundle in zip(names, bundles):
        describe = getattr(bundle.client, "describe_model", None)
        if describe is None:
            print("error: this client does not support --preflight", file=sys.stderr)
            return EXIT_USAGE
        try:
            info = describe(model)
        except LLMError as exc:
            print(f"error: preflight failed: {describe_exception(exc, bundle.secrets)}",
                  file=sys.stderr)
            return EXIT_FATAL
        print(json.dumps({"judge": judge_id, "model": model, "family": family, "info": info},
                         sort_keys=True))
        if not info.get("found"):
            missing.append(model)
        if info.get("supports_response_schema") is not True:
            print(f"note: {model} does not report supports_response_schema; "
                  "the plain-JSON path will be used", file=sys.stderr)
    if missing:
        print(f"error: model not found: {', '.join(missing)}", file=sys.stderr)
        return EXIT_FATAL
    return EXIT_OK


def run(args: argparse.Namespace, ctx: Context) -> int:
    if args.preflight and args.dry_run:
        raise ValueError("--preflight cannot be combined with --dry-run")
    if args.stage is None and not args.preflight:
        raise ValueError("--stage is required")
    names = _judge_ids_and_names(args)
    extra_body = factory.parse_extra_body(args.extra_body)
    if args.preflight:
        return _preflight(args, ctx, names)
    rubric = judges.load_rubric(args.rubric_file)
    if args.stage == "test":
        _check_frozen(args.out_dir, rubric)
    runners = [] if args.dry_run else _build_judges(args, ctx, names, rubric, extra_body)
    backend = runners[0].info.backend if runners else args.backend or DEFAULT_BACKEND
    infos = [JudgeInfo(id=i, model=m, family=f, backend=backend) for i, m, f in names]
    root = artifacts.resolve_artifacts_root(args.artifacts_dir)
    checked = sorted({*defaults.DEFAULT_ALSO_USED_MODELS,
                      *loading.pipeline_models(root, args.also_used_model)})
    judges.check_independence(infos, checked)
    recorded = _check_manifest(args.out_dir, infos)
    pairs = _read_pair_texts(args, args.stage)
    if args.dry_run:
        print(json.dumps({
            "stage": args.stage, "pairs": len(pairs), "calls": 2 * len(pairs),
            "already_done": _already_done(args.out_dir, rubric, pairs),
        }, sort_keys=True))
        return EXIT_OK
    if not recorded:
        layout.update_manifest(args.out_dir, {
            "judges": [vars(info) for info in infos], "rubric_hash": rubric.hash,
        })
    summary, failed, fatal = _judge_all(args, runners, pairs, rubric)
    report: dict = {"stage": args.stage, "judges": summary, "checked_against": checked}
    if not fatal:
        report["labeled_pairs"] = cmd_rubric.rebuild_labels(args.out_dir, rubric)
    print(json.dumps(report, sort_keys=True))
    return EXIT_FATAL if fatal else EXIT_FAILED_ITEMS if failed else EXIT_OK

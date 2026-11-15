"""evaluate sample: write the posting pool, the query sets and the run manifest."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from getwork import fileio
from getwork.evaluation import args as eval_args
from getwork.evaluation import layout, loading, sampling
from getwork.evaluation.args import EXIT_OK, EXIT_USAGE, Context

COMMANDS = ("sample",)
OUTPUT_FILES = (layout.MANIFEST_FILE, layout.SETS_FILE, layout.POOL_POSTINGS_FILE)


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number < 0:
        raise argparse.ArgumentTypeError(f"must not be negative: {value}")
    return number


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("sample", help="sample the pool and the dev, test and human sets")
    eval_args.add_common_args(parser)
    defaults = sampling.SampleParams()
    parser.add_argument("--seed", type=int, default=defaults.seed)
    for field in ("pool_cap", "dev_resumes", "dev_postings", "test_resumes", "test_postings",
                  "human_resumes", "human_postings"):
        parser.add_argument(
            f"--{field.replace('_', '-')}", type=_non_negative_int, default=getattr(defaults, field)
        )
    parser.add_argument("--force", action="store_true", help="overwrite an earlier sample")


def _params(args: argparse.Namespace) -> sampling.SampleParams:
    names = sampling.SampleParams.__dataclass_fields__
    return sampling.SampleParams(**{name: getattr(args, name) for name in names})


def _summary(sets: dict) -> dict:
    counts = {
        stage: {side: len(ids) for side, ids in sides.items()}
        for stage, sides in sets["stages"].items()
    }
    keys = ("n_pool", "n_resumes_labeled", "n_resumes_unlabeled")
    return {"stages": counts, **{key: sets[key] for key in keys}}


def _existing(out_dir: Path) -> list[str]:
    return [name for name in OUTPUT_FILES if (out_dir / name).exists()]


def run(args: argparse.Namespace, ctx: Context) -> int:
    params = _params(args)
    loaded = loading.load_eval_inputs(args.artifacts_dir, args.processed_dir)
    sets = sampling.build_sets(
        resume_labels=loaded.first_labels, posting_ids=list(loaded.data.postings), params=params
    )
    existing = _existing(args.out_dir)
    if existing and not args.force:
        print(f"error: {existing[0]} already exists in {args.out_dir}; use --force to "
              f"sample again", file=sys.stderr)
        return EXIT_USAGE
    pool = sets.pop("pool")
    fileio.write_json_atomic(args.out_dir / layout.POOL_POSTINGS_FILE, {"ids": pool})
    fileio.write_json_atomic(args.out_dir / layout.SETS_FILE, sets)
    layout.update_manifest(args.out_dir, {
        "schema": sampling.SCHEMA_VERSION,
        "seed": params.seed,
        "artifacts_dir": str(loaded.root),
        "processed_dir": str(args.processed_dir),
        "params": asdict(params),
        "stamps": layout.stamps_of(loaded.data),
    })
    print(json.dumps(_summary(sets), sort_keys=True))
    return EXIT_OK

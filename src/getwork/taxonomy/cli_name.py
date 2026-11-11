"""The name and build subcommands (TX-6, TX-7): editable cluster names, then taxonomy.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from getwork import fileio, paths
from getwork.llm import factory as llm_factory
from getwork.llm.client import FatalLLMError, LLMClient, redact_secrets
from getwork.taxonomy import build, cli_common, model, naming, overlap
from getwork.taxonomy.cli_common import SelectionState

NAMES_FILE = "cluster_names.csv"
TAXONOMY_FILE = "taxonomy.json"
DEFAULT_TOP_MEMBERS = 25
DEFAULT_NAME_WORKERS: int = 1
DEFAULT_OVERLAP_ROUNDS: int = 2
DEFAULT_MAX_TOKENS = 1024
# Same choices and defaults as the structurer (structure/cli.py): "none" for Voyager, omitted for local.
REASONING_CHOICES = ["none", "low", "medium", "high", "omit"]
LOCAL_BACKEND = "local"
VOYAGER_REASONING_EFFORT = "none"
EXAMPLE_MEMBERS = 5
METHODS = ("heuristic", "llm")
LLM_ROW = "llm"
HEURISTIC_ROW = "heuristic"
FALLBACK_ROW = "heuristic_fallback"
OVERLAP_STATUS = "overlap"


def add_parsers(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("name", help="write cluster_names.csv for review")
    parser.add_argument("--method", choices=METHODS, default="heuristic")
    parser.add_argument("--top-members", type=cli_common.positive_int, default=DEFAULT_TOP_MEMBERS)
    parser.add_argument("--max-tokens", type=cli_common.positive_int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--reasoning-effort", choices=REASONING_CHOICES, default=None)
    parser.add_argument("--workers", type=cli_common.positive_int, default=None)
    parser.add_argument("--overlap-rounds", type=cli_common.nonnegative_int, default=None)
    parser.add_argument("--overlap-judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    llm_factory.add_llm_args(parser)
    parser = sub.add_parser("build", help="write taxonomy.json from the edited names")
    parser.add_argument("--names", type=Path, default=None)
    parser.add_argument("--allow-member-names", action="store_true")


def _members_by_cluster(state: SelectionState) -> dict[int, list[str]]:
    members: dict[int, list[str]] = {}
    for skill, label in zip(state.skills, (int(x) for x in state.labels)):
        if label >= 0:
            members.setdefault(label, []).append(skill)
    return members


def _stderr_log(secrets: Sequence[str]) -> Callable[[str], None]:
    def log(message: str) -> None:
        print(redact_secrets(message, secrets), file=sys.stderr)

    return log


def _resolve_reasoning_effort(args: argparse.Namespace, backend: str) -> str | None:
    if args.reasoning_effort is not None:
        return None if args.reasoning_effort == "omit" else args.reasoning_effort
    return None if backend == LOCAL_BACKEND else VOYAGER_REASONING_EFFORT


def _llm_names(
    args: argparse.Namespace,
    *,
    bundle: llm_factory.ClientBundle,
    members: Mapping[int, list[str]],
    extra_body: dict | None,
) -> tuple[naming.NameResult, overlap.ReviewResult]:
    cfg = naming.NameConfig(
        model=bundle.model,
        backend=bundle.backend,
        top_members=args.top_members,
        max_tokens=args.max_tokens,
        reasoning_effort=_resolve_reasoning_effort(args, bundle.backend),
        extra_body=extra_body,
    )
    workers = args.workers or DEFAULT_NAME_WORKERS
    log = _stderr_log(bundle.secrets)
    rounds = DEFAULT_OVERLAP_ROUNDS if args.overlap_rounds is None else args.overlap_rounds
    try:
        named = naming.name_clusters(
            bundle.client, cfg, members, secrets=bundle.secrets, workers=workers, log=log
        )
        review = overlap.review_overlaps(
            bundle.client,
            cfg,
            overlap.ReviewConfig(rounds=rounds, judge=args.overlap_judge),
            named.names,
            {i: members[i] for i in named.names},
            workers=workers,
            log=log,
        )
    except FatalLLMError as exc:
        raise cli_common.FatalError(redact_secrets(str(exc), bundle.secrets)) from exc
    return named, review


def _prompt_version(*, from_llm: bool, renamed: bool) -> str:
    if not from_llm:
        return ""
    if renamed:
        return f"{naming.NAME_PROMPT_VERSION}+{overlap.RENAME_PROMPT_VERSION}"
    return naming.NAME_PROMPT_VERSION


def _name_rows(
    *,
    members: Mapping[int, list[str]],
    llm: naming.NameResult | None,
    bundle: llm_factory.ClientBundle | None,
    review: overlap.ReviewResult | None = None,
) -> list[list[object]]:
    names: dict[int, str] = {}
    methods: dict[int, str] = {}
    for cluster_id, skills in members.items():
        if llm is not None and cluster_id in llm.names:
            reviewed = review.names if review is not None else llm.names
            names[cluster_id] = reviewed.get(cluster_id, llm.names[cluster_id])
            methods[cluster_id] = LLM_ROW
        else:
            names[cluster_id] = naming.heuristic_name(skills)
            methods[cluster_id] = HEURISTIC_ROW if llm is None else FALLBACK_ROW
    status = build.name_status(names, members)
    renamed = review.renamed if review is not None else {}
    for cluster_id in review.unresolved if review is not None else ():
        if status[cluster_id] == "ok":
            status[cluster_id] = OVERLAP_STATUS
    rows = []
    for cluster_id in sorted(members):
        from_llm = methods[cluster_id] == LLM_ROW and bundle is not None
        rows.append(
            [
                cluster_id,
                names[cluster_id],
                len(members[cluster_id]),
                "; ".join(members[cluster_id][:EXAMPLE_MEMBERS]),
                methods[cluster_id],
                status[cluster_id],
                bundle.model if from_llm else "",
                bundle.backend if from_llm else "",
                _prompt_version(from_llm=from_llm, renamed=cluster_id in renamed),
            ]
        )
    return rows


def run_name(
    args: argparse.Namespace, llm_client: LLMClient | None, environ: Mapping[str, str]
) -> int:
    bundle = None
    extra_body = None
    if args.method == "llm":
        extra_body = llm_factory.parse_extra_body(args.extra_body)
        bundle = llm_factory.make_client(
            backend=args.backend,
            model=args.model,
            env_file=Path(args.env_file),
            environ=environ,
            client=llm_client,
        )
    root = args.artifacts_dir
    target = paths.taxonomy_dir(root) / NAMES_FILE
    if target.exists() and not args.force:
        raise ValueError(f"{target} already exists; use --force to overwrite your edits")
    members = _members_by_cluster(cli_common.load_selection_state(root))
    llm, review = None, None
    if bundle is not None:
        llm, review = _llm_names(args, bundle=bundle, members=members, extra_body=extra_body)
    rows = _name_rows(members=members, llm=llm, bundle=bundle, review=review)
    fileio.write_csv_atomic(target, build.NAMES_COLUMNS, rows)
    n_fallback = sum(1 for row in rows if row[4] == FALLBACK_ROW)
    summary = {
        "names_file": str(target),
        "method": args.method,
        "n_clusters": len(rows),
        "n_fallback": n_fallback,
        "n_duplicate": sum(1 for row in rows if row[5] == "duplicate"),
        "n_member_list": sum(1 for row in rows if row[5] == "member_list"),
        "n_renamed": len(review.renamed) if review is not None else 0,
        "n_overlap": sum(1 for row in rows if row[5] == OVERLAP_STATUS),
    }
    print(json.dumps(summary))
    return cli_common.EXIT_FALLBACK if n_fallback else cli_common.EXIT_OK


def _refuse_member_list_names(
    names: Mapping[int, str], members: Mapping[int, list[str]], *, allow: bool
) -> None:
    bad = sorted(i for i, n in names.items() if model.member_list_reason(n, members.get(i, ())))
    if bad and not allow:
        raise ValueError(
            f"names for cluster ids {bad} are lists of member skills; "
            "write a concept label or pass --allow-member-names"
        )


def _warn_about_overlap(names: Mapping[int, str], members: Mapping[int, list[str]]) -> int:
    """Rules-only overlap warnings on stderr; member-list names are ignored, nothing is refused."""
    concept_names = {
        i: n for i, n in names.items() if not model.member_list_reason(n, members.get(i, ()))
    }
    groups = overlap.rule_groups(concept_names)
    for group in groups:
        ids = ", ".join(map(str, group.ids))
        print(
            f"warning: clusters {ids} have overlapping names ({group.reason}); "
            "rename them in cluster_names.csv or ignore this",
            file=sys.stderr,
        )
    return len(groups)


def run_build(args: argparse.Namespace) -> int:
    root = args.artifacts_dir
    out = paths.taxonomy_dir(root)
    names_path = args.names or out / NAMES_FILE
    names = build.read_names(names_path)
    state = cli_common.load_selection_state(root)
    members = _members_by_cluster(state)
    _refuse_member_list_names(names, members, allow=args.allow_member_names)
    n_overlap_groups = _warn_about_overlap(names, members)
    summary = cli_common.read_counts_summary(root)
    vectors = cli_common.read_skill_vectors(root)
    if vectors.meta.get("selected_skills_version") != summary["selected_skills_version"]:
        raise ValueError("skill vectors are stale; run taxonomy embed and taxonomy grid again")
    meta = {
        "selection": state.selection,
        "embedding": vectors.meta["embedding"],
        "names_file_sha256": hashlib.sha256(names_path.read_bytes()).hexdigest(),
    }
    taxonomy = build.build_taxonomy_from_labels(
        state.skills,
        state.labels,
        names,
        alias_table_version=summary["alias_table_version"],
        meta=meta,
    )
    target = out / TAXONOMY_FILE
    model.save_taxonomy(target, taxonomy)
    print(
        json.dumps(
            {
                "taxonomy_file": str(target),
                "version": taxonomy.version,
                "alias_table_version": taxonomy.alias_table_version,
                "n_clusters": len(taxonomy.clusters),
                "n_noise": len(taxonomy.noise),
                "n_overlap_groups": n_overlap_groups,
                "config_id": state.selection["config_id"],
            }
        )
    )
    return cli_common.EXIT_OK

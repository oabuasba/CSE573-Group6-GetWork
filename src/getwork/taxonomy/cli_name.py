"""The name and build subcommands (TX-6, TX-7): editable cluster names, then taxonomy.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from getwork import fileio, paths
from getwork.llm import factory as llm_factory
from getwork.llm.client import FatalLLMError, LLMClient, redact_secrets
from getwork.taxonomy import build, cli_common, model, naming
from getwork.taxonomy.cli_common import SelectionState

NAMES_FILE = "cluster_names.csv"
TAXONOMY_FILE = "taxonomy.json"
DEFAULT_TOP_MEMBERS = 25
EXAMPLE_MEMBERS = 5
METHODS = ("heuristic", "llm")
LLM_ROW = "llm"
HEURISTIC_ROW = "heuristic"
FALLBACK_ROW = "heuristic_fallback"


def add_parsers(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("name", help="write cluster_names.csv for review")
    parser.add_argument("--method", choices=METHODS, default="heuristic")
    parser.add_argument("--top-members", type=cli_common.positive_int, default=DEFAULT_TOP_MEMBERS)
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


def _llm_names(
    args: argparse.Namespace,
    *,
    bundle: llm_factory.ClientBundle,
    members: Mapping[int, list[str]],
    extra_body: dict | None,
) -> naming.NameResult:
    cfg = naming.NameConfig(
        model=bundle.model,
        backend=bundle.backend,
        top_members=args.top_members,
        extra_body=extra_body,
    )
    try:
        return naming.name_clusters(bundle.client, cfg, members, secrets=bundle.secrets)
    except FatalLLMError as exc:
        raise cli_common.FatalError(redact_secrets(str(exc), bundle.secrets)) from exc


def _name_rows(
    *,
    members: Mapping[int, list[str]],
    llm: naming.NameResult | None,
    bundle: llm_factory.ClientBundle | None,
) -> list[list[object]]:
    names: dict[int, str] = {}
    methods: dict[int, str] = {}
    for cluster_id, skills in members.items():
        if llm is not None and cluster_id in llm.names:
            names[cluster_id], methods[cluster_id] = llm.names[cluster_id], LLM_ROW
        else:
            names[cluster_id] = naming.heuristic_name(skills)
            methods[cluster_id] = HEURISTIC_ROW if llm is None else FALLBACK_ROW
    status = build.name_status(names, members)
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
                naming.NAME_PROMPT_VERSION if from_llm else "",
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
    llm = None
    if bundle is not None:
        llm = _llm_names(args, bundle=bundle, members=members, extra_body=extra_body)
    rows = _name_rows(members=members, llm=llm, bundle=bundle)
    fileio.write_csv_atomic(target, build.NAMES_COLUMNS, rows)
    n_fallback = sum(1 for row in rows if row[4] == FALLBACK_ROW)
    summary = {
        "names_file": str(target),
        "method": args.method,
        "n_clusters": len(rows),
        "n_fallback": n_fallback,
        "n_duplicate": sum(1 for row in rows if row[5] == "duplicate"),
        "n_member_list": sum(1 for row in rows if row[5] == "member_list"),
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


def run_build(args: argparse.Namespace) -> int:
    root = args.artifacts_dir
    out = paths.taxonomy_dir(root)
    names_path = args.names or out / NAMES_FILE
    names = build.read_names(names_path)
    state = cli_common.load_selection_state(root)
    _refuse_member_list_names(names, _members_by_cluster(state), allow=args.allow_member_names)
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
                "config_id": state.selection["config_id"],
            }
        )
    )
    return cli_common.EXIT_OK

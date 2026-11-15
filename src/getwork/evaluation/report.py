"""Markdown rendering of the evaluation results; pure, no clocks, no file access."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from getwork.evaluation.metrics import METRICS
from getwork.retrieval.inputs import DIRECTIONS, SYSTEM_IDS

NOT_RUN = "_not run yet_"
POOL_PHRASE = "software and adjacent IT postings"
JUDGE_GROUP = "judge@test"
HUMAN_SUFFIX = "@human"
GAP_NAMES = ("B2-B1+", "Full-B2", "Full-B1+")
PRIMARY_METRIC = "nDCG@10"
P_FLOOR = 0.001

CAVEATS: tuple[str, ...] = (
    "A pair is relevant (label 2) only when the lower of the two labels, resume to posting and "
    "posting to resume, is 2: the lower-label rule is deliberately strict.",
    "With three systems Kendall's tau between system orders can only be 1, 1/3, -1/3 or -1, so "
    "the gap sizes are reported beside it.",
    "There is no multiplicity correction: three gaps, four metrics and two directions are "
    "tested separately and the p-values are not adjusted.",
    "The relevance labels come from LLM judges, not from people; their agreement with humans is "
    "checked on a small set only.",
    "The human check set is small (tens of pairs), so its kappa and tau values are rough.",
    f"The posting pool is {POOL_PHRASE}, a seeded sample of the tech postings table, not all "
    "postings.",
)
B2_CAVEAT = (
    "B2 text keeps the Full grouping order (b2_order = grouped, architecture Q-10), so part of "
    "the taxonomy effect can appear in B2; rebuild documents with --b2-order record for a "
    "cleaner ablation."
)


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _p_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"<{P_FLOOR}" if value < P_FLOOR else f"{value:.3f}"


def _cell(entry: Mapping) -> str:
    low, high = entry["ci"]
    return f"{_num(entry['mean'])} [{_num(low)}, {_num(high)}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return [*lines, ""]


def _run_section(manifest: Mapping) -> list[str]:
    stamps = manifest.get("stamps", {})
    docs = stamps.get("documents", {})
    params = manifest.get("params", {})
    lines = [
        f"- Seed: {manifest.get('seed', 'n/a')}",
        f"- Pool: {POOL_PHRASE}, capped at {params.get('pool_cap', 'n/a')} by a seeded sample",
        f"- Resumes: {stamps.get('n_resumes', 'n/a')}; postings: {stamps.get('n_postings', 'n/a')}",
        f"- Taxonomy version: {docs.get('taxonomy_version', 'n/a')}",
        f"- Relabel mode: {docs.get('relabel_mode', 'n/a')}",
        f"- B2 text order (b2_order): {docs.get('b2_order', 'n/a')}",
    ]
    for stage, weights in sorted(manifest.get("weights", {}).items()):
        parts = ", ".join(f"{k} {_num(weights.get(k))}" for k in ("skills", "title", "level"))
        lines.append(f"- B1+ weights ({stage}): {parts}; source {weights.get('source', 'n/a')}")
    sizes = ", ".join(f"{k} {v}" for k, v in sorted(params.items()) if k != "pool_cap")
    return [*lines, f"- Parameters: {sizes or 'n/a'}", ""]


def _kappa_of(agreement: Mapping | None, stage: str) -> str:
    block = ((agreement or {}).get("judges") or {}).get(stage)
    return "n/a" if not block else _num(block.get("kappa"))


def _judges_section(
    manifest: Mapping, agreement: Mapping | None, frozen: Mapping | None
) -> list[str]:
    if frozen is None:
        return [NOT_RUN, ""]
    rows = [
        [j.get("id", "n/a"), j.get("model", "n/a"), j.get("family", "n/a"),
         str(j.get("truncated", "n/a"))]
        for j in manifest.get("judges", [])
    ]
    return [
        f"- Rubric hash: {frozen.get('rubric_hash', 'n/a')}",
        f"- Judge kappa, dev pairs: {_kappa_of(agreement, 'dev')}",
        f"- Judge kappa, test pairs: {_kappa_of(agreement, 'test')}",
        "",
        *_table(["Judge", "Model", "Family", "Truncated pairs"], rows),
    ]


def _ranking_section(group: Mapping) -> list[str]:
    lines: list[str] = []
    for direction in DIRECTIONS:
        block = group["directions"][direction]
        rows = [[s, *(_cell(block["systems"][s][m]) for m in METRICS)] for s in SYSTEM_IDS]
        lines += [f"### {direction} ({block['n_queries']} queries)", ""]
        lines += _table(["System", *METRICS], rows)
    return lines


def _gap_section(group: Mapping) -> list[str]:
    lines: list[str] = []
    for direction in DIRECTIONS:
        gaps = group["directions"][direction]["gaps"]
        rows = [
            [name, m, _num(gaps[name][m]["diff"]),
             f"[{_num(gaps[name][m]['ci'][0])}, {_num(gaps[name][m]['ci'][1])}]",
             _p_value(gaps[name][m]["p"])]
            for name in GAP_NAMES for m in METRICS
        ]
        lines += [f"### {direction}", ""]
        lines += _table(["Gap", "Metric", "Diff", "95% CI", "p"], rows)
    return lines


def _human_group_rows(metrics: Mapping | None) -> list[list[str]]:
    groups = (metrics or {}).get("groups", {})
    rows = []
    for key in sorted(k for k in groups if k.endswith(HUMAN_SUFFIX)):
        for direction in DIRECTIONS:
            block = groups[key]["directions"][direction]
            means = [_num(block["systems"][s][PRIMARY_METRIC]["mean"]) for s in SYSTEM_IDS]
            gap = block["gaps"]["Full-B2"][PRIMARY_METRIC]["diff"]
            rows.append([key, direction, *means, _num(gap)])
    return rows


def _tau_lines(tau: Mapping | None) -> list[str]:
    if not tau:
        return ["- Kendall tau (nDCG@10): n/a"]
    lines = []
    for direction, values in sorted(tau.get(PRIMARY_METRIC, {}).items()):
        text = ", ".join(f"{k} {_num(v)}" for k, v in sorted(values.items()))
        lines.append(f"- Kendall tau-b of nDCG@10 system orders, {direction}: {text}")
    return lines


def _human_section(agreement: Mapping | None, metrics: Mapping | None) -> list[str]:
    human = (agreement or {}).get("human")
    if not human:
        return [NOT_RUN, ""]
    fallback = human.get("fallback") or {}
    lines = [
        f"- Half: {human.get('half', 'n/a')}; pairs: {human.get('n_pairs', 'n/a')}; "
        f"authors: {', '.join(human.get('authors', [])) or 'n/a'}",
        f"- Author kappa: {_num(human.get('author_kappa'))}",
        f"- Human-judge kappa: {_num(human.get('human_judge_kappa'))}",
        f"- Decision: {fallback.get('decision', 'n/a')} "
        f"(threshold {_num(fallback.get('threshold'))})",
        *_tau_lines(human.get("tau")),
        "",
    ]
    rows = _human_group_rows(metrics)
    if rows:
        header = ["Group", "Direction", *(f"{s} nDCG@10" for s in SYSTEM_IDS), "Full-B2 gap"]
        lines += _table(header, rows)
    return lines


def _ari_section(ari: Mapping | None) -> list[str]:
    if ari is None:
        return [NOT_RUN, ""]
    taxonomy, gap = ari.get("taxonomy", {}), ari.get("vocabulary_gap", {})
    rows = [
        [name, str(block.get("dims", "n/a")), _num(block.get("ari_mean")),
         _num(block.get("ari_sd"))]
        for name, block in (("Cluster profiles", ari.get("clusters", {})),
                            ("Raw skills", ari.get("raw_skills", {})))
    ]
    return [
        f"- k: {ari.get('k', 'n/a')}; seeds: {len(ari.get('seeds', []))}; "
        f"labeled resumes: {ari.get('n_resumes', 'n/a')} "
        f"({ari.get('n_unlabeled_excluded', 'n/a')} unlabeled excluded)",
        f"- Taxonomy: {taxonomy.get('config_id', 'n/a')}, {taxonomy.get('n_clusters', 'n/a')} "
        f"clusters, silhouette {_num(taxonomy.get('silhouette'))}, "
        f"noise fraction {_num(taxonomy.get('noise_fraction'))}",
        f"- Vocabulary in one corpus only: {gap.get('n_one_corpus', 'n/a')} of "
        f"{gap.get('n_retained', 'n/a')} ({_num(gap.get('fraction_one_corpus'))})",
        "",
        *_table(["Profile", "Dims", "ARI mean", "ARI sd"], rows),
    ]


def _caveat_lines(manifest: Mapping) -> list[str]:
    lines = list(CAVEATS)
    b2_order = manifest.get("stamps", {}).get("documents", {}).get("b2_order")
    if b2_order == "grouped":
        lines.append(B2_CAVEAT)
    return [*(f"- {line}" for line in lines), ""]


def render_report(
    *, manifest: Mapping, metrics: Mapping | None, agreement: Mapping | None,
    ari: Mapping | None, frozen: Mapping | None,
) -> str:
    group = ((metrics or {}).get("groups") or {}).get(JUDGE_GROUP)
    ranking = _ranking_section(group) if group else [NOT_RUN, ""]
    gaps = _gap_section(group) if group else [NOT_RUN, ""]
    sections = [
        ("# Evaluation report", []),
        ("## Run", _run_section(manifest)),
        ("## Judges", _judges_section(manifest, agreement, frozen)),
        ("## Ranking quality (judge labels, test queries)", ranking),
        ("## System gaps", gaps),
        ("## Human check", _human_section(agreement, metrics)),
        ("## Taxonomy validity (ARI)", _ari_section(ari)),
        ("## Caveats", _caveat_lines(manifest)),
    ]
    lines: list[str] = []
    for heading, body in sections:
        lines += [heading, "", *body]
    return "\n".join(lines).rstrip("\n") + "\n"

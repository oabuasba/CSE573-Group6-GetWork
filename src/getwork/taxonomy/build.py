"""Names file (the human-edited cluster_names.csv) and taxonomy assembly."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from getwork.taxonomy import model

NAMES_COLUMNS: tuple[str, ...] = (
    "cluster_id",
    "name",
    "size",
    "top_members",
    "method",
    "status",
    "model",
    "backend",
    "prompt_version",
)
NOISE_LABEL = -1


def _ids_by_folded_name(names: Mapping[int, str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for cluster_id, name in names.items():
        groups[name.casefold()].append(cluster_id)
    return groups


def name_status(names: Mapping[int, str]) -> dict[int, str]:
    duplicated = {i for ids in _ids_by_folded_name(names).values() if len(ids) > 1 for i in ids}
    return {i: "duplicate" if i in duplicated else "ok" for i in names}


def _parse_id(raw: str, line: int) -> int:
    try:
        return int(raw.strip())
    except ValueError:
        raise ValueError(f"names file line {line}: cluster_id is not an integer") from None


def _read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not {"cluster_id", "name"} <= set(reader.fieldnames or ()):
                raise ValueError("names file needs cluster_id and name columns")
            return list(reader)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"names file not found: {path} (run taxonomy name)") from e


def _check_names(names: Mapping[int, str]) -> None:
    bad = []
    for cluster_id, name in names.items():
        try:
            model.check_name(name)
        except ValueError:
            bad.append(cluster_id)
    if bad:
        raise ValueError(f"invalid names for cluster ids {sorted(bad)}")
    clashes = sorted(
        i for ids in _ids_by_folded_name(names).values() if len(ids) > 1 for i in ids
    )
    if clashes:
        raise ValueError(f"duplicate names for cluster ids {clashes}")


def read_names(path: Path) -> dict[int, str]:
    rows = _read_rows(path)
    names: dict[int, str] = {}
    repeated: list[int] = []
    for line, row in enumerate(rows, start=2):
        cluster_id = _parse_id(row["cluster_id"], line)
        if cluster_id in names:
            repeated.append(cluster_id)
        names[cluster_id] = (row["name"] or "").strip()
    if repeated:
        raise ValueError(f"repeated cluster ids {sorted(set(repeated))}")
    _check_names(names)
    return names


def build_taxonomy_from_labels(
    skills: Sequence[str],
    labels: np.ndarray,
    names: Mapping[int, str],
    *,
    alias_table_version: str,
    meta: Mapping[str, object],
) -> model.Taxonomy:
    label_list = [int(x) for x in labels]
    if len(label_list) != len(skills):
        raise ValueError("skills and labels must have the same length")
    members: dict[int, list[str]] = defaultdict(list)
    for skill, label in zip(skills, label_list):
        members[label].append(skill)
    noise = members.pop(NOISE_LABEL, [])
    missing, extra = sorted(set(members) - set(names)), sorted(set(names) - set(members))
    if missing or extra:
        raise ValueError(
            f"names and labels disagree: no name for cluster ids {missing}, "
            f"no cluster for cluster ids {extra}"
        )
    clusters = [
        model.Cluster(id=i, name=names[i], skills=tuple(members[i])) for i in sorted(members)
    ]
    return model.make_taxonomy(clusters, noise, alias_table_version, meta)

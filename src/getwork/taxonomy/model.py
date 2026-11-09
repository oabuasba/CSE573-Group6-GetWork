"""Skill taxonomy: named clusters of canonical skills plus a noise list."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from getwork import fileio
from getwork.alias import normalize

OTHER: str = "Other"
MAX_NAME_CHARS: int = 40
FORBIDDEN_NAME_CHARS: str = ":;,\n\r"
VERSION_PREFIX = "sha256:"
VERSION_HEX_CHARS = 12


def check_name(name: str) -> str:
    """Strip and validate a cluster name; ValueError naming the broken rule."""
    name = name.strip()
    if not name:
        raise ValueError("cluster name must not be empty")
    if len(name) > MAX_NAME_CHARS:
        raise ValueError(f"cluster name must be at most {MAX_NAME_CHARS} characters")
    if any(ch in name for ch in FORBIDDEN_NAME_CHARS):
        raise ValueError("cluster name must not contain ':', ';', ',' or a line break")
    if name.casefold() == OTHER.casefold():
        raise ValueError(f"cluster name must not be {OTHER!r}")
    return name


@dataclass(frozen=True)
class Cluster:
    id: int
    name: str
    skills: tuple[str, ...]


@dataclass(frozen=True)
class Taxonomy:
    version: str
    alias_table_version: str
    clusters: tuple[Cluster, ...]
    noise: tuple[str, ...]
    meta: Mapping[str, object]
    _index: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def lookup(self, skill: str) -> str | None:
        """Cluster name, OTHER for noise, None when the skill is not in the taxonomy."""
        return self._index.get(normalize.normalize_key(skill))

    def is_known(self, skill: str) -> bool:
        return self.lookup(skill) is not None

    def cluster_names(self) -> list[str]:
        return [c.name for c in self.clusters]

    def examples(self, n: int = 5) -> dict[str, list[str]]:
        return {c.name: list(c.skills[:n]) for c in self.clusters}


def taxonomy_version(clusters: Sequence[Cluster], noise: Sequence[str]) -> str:
    payload = {
        "clusters": [[c.id, c.name, list(c.skills)] for c in sorted(clusters, key=_cluster_id)],
        "noise": sorted(noise),
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return VERSION_PREFIX + digest.hexdigest()[:VERSION_HEX_CHARS]


def _cluster_id(cluster: Cluster) -> int:
    return cluster.id


def _build_index(clusters: Sequence[Cluster], noise: Sequence[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    members = [(s, c.name) for c in clusters for s in c.skills] + [(s, OTHER) for s in noise]
    for skill, owner in members:
        key = normalize.normalize_key(skill)
        if key in index:
            raise ValueError(f"skill appears more than once in the taxonomy: {skill!r}")
        index[key] = owner
    return index


def make_taxonomy(
    clusters: Sequence[Cluster],
    noise: Sequence[str],
    alias_table_version: str,
    meta: Mapping[str, object],
) -> Taxonomy:
    """Validate, sort clusters by id and compute the version."""
    if not clusters:
        raise ValueError("a taxonomy needs at least one cluster")
    names = [check_name(c.name) for c in clusters]
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError("cluster names must be unique ignoring case")
    if len({c.id for c in clusters}) != len(clusters):
        raise ValueError("cluster ids must be unique")
    ordered = tuple(
        Cluster(id=c.id, name=n, skills=tuple(c.skills))
        for c, n in sorted(zip(clusters, names), key=lambda pair: pair[0].id)
    )
    noise_t = tuple(noise)
    tax = Taxonomy(
        version=taxonomy_version(ordered, noise_t),
        alias_table_version=alias_table_version,
        clusters=ordered,
        noise=noise_t,
        meta=meta,
    )
    return replace(tax, _index=_build_index(ordered, noise_t))


def save_taxonomy(path: Path, tax: Taxonomy) -> None:
    fileio.write_json_atomic(
        path,
        {
            "version": tax.version,
            "alias_table_version": tax.alias_table_version,
            "n_skills": sum(len(c.skills) for c in tax.clusters) + len(tax.noise),
            "n_clusters": len(tax.clusters),
            "n_noise": len(tax.noise),
            "clusters": [
                {"id": c.id, "name": c.name, "skills": list(c.skills)} for c in tax.clusters
            ],
            "noise": list(tax.noise),
            "meta": dict(tax.meta),
        },
    )


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _check_file_types(raw: object) -> None:
    if not isinstance(raw, dict) or not isinstance(raw.get("clusters"), list):
        raise TypeError("expected an object with a clusters list")
    if not _is_str_list(raw.get("noise")):
        raise TypeError("noise must be a list of strings")
    for c in raw["clusters"]:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str):
            raise TypeError("cluster name must be a string")
        if not isinstance(c.get("id"), int) or not _is_str_list(c.get("skills")):
            raise TypeError("cluster id must be an int and skills a list of strings")


def load_taxonomy(path: Path) -> Taxonomy:
    """Load and re-validate; ValueError if the stored version is not the recomputed one."""
    try:
        raw = fileio.read_json(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"taxonomy file not found: {path} (run taxonomy build)") from e
    try:
        _check_file_types(raw)
        clusters = [
            Cluster(id=c["id"], name=c["name"], skills=tuple(c["skills"]))
            for c in raw["clusters"]
        ]
        tax = make_taxonomy(
            clusters, raw["noise"], raw["alias_table_version"], raw.get("meta", {})
        )
        stored = raw["version"]
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"taxonomy file {path} is malformed: {e!r}") from e
    if stored != tax.version:
        raise ValueError(f"taxonomy file {path} version {stored} does not match {tax.version}")
    return tax

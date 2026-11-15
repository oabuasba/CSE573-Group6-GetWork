"""Taxonomy validity: do resume profiles built on clusters recover occupation labels better than raw skills."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pyarrow as pa

from getwork.taxonomy.model import OTHER


def cluster_profiles(docs: Mapping[str, dict], cluster_names: Sequence[str]) -> np.ndarray:
    """Per-document count of skills in each cluster, ignoring Other; docs order, float32."""
    column = {name: i for i, name in enumerate(cluster_names)}
    matrix = np.zeros((len(docs), len(cluster_names)), dtype=np.float32)
    for row, doc in enumerate(docs.values()):
        for item in doc["skill_clusters"]:
            cluster = item["cluster"]
            if cluster == OTHER:
                continue
            if cluster not in column:
                raise ValueError(f"unknown cluster {cluster!r}")
            matrix[row, column[cluster]] += 1
    return matrix


def skill_profiles(docs: Mapping[str, dict], skills: Sequence[str]) -> np.ndarray:
    """1 where the exact skill string is among the document's skills, else 0; float32."""
    column = {skill: i for i, skill in enumerate(skills)}
    matrix = np.zeros((len(docs), len(skills)), dtype=np.float32)
    for row, doc in enumerate(docs.values()):
        for skill in doc["skills"]:
            if skill in column:
                matrix[row, column[skill]] = 1
    return matrix


def ari_over_seeds(
    X: np.ndarray, labels: Sequence[str], *, k: int, seeds: Sequence[int]
) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    scores = []
    for seed in seeds:
        assignment = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        scores.append(float(adjusted_rand_score(labels, assignment)))
    return {
        "dims": int(X.shape[1]),
        "ari_by_seed": scores,
        "ari_mean": float(np.mean(scores)) if scores else None,
        "ari_sd": float(np.std(scores)) if scores else None,
    }


def vocabulary_gap(skill_counts: pa.Table) -> dict:
    """Share of selected skills that occur in only one of the two corpora."""
    rows = [r for r in skill_counts.to_pylist() if r["selected"]]
    one_corpus = sum(1 for r in rows if r["df_resumes"] == 0 or r["df_postings"] == 0)
    return {
        "n_retained": len(rows),
        "n_one_corpus": one_corpus,
        "fraction_one_corpus": one_corpus / len(rows) if rows else 0.0,
    }

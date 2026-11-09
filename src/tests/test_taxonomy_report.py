import json
from pathlib import Path

import numpy as np
import pytest

from getwork.taxonomy import grid, report

pytestmark = pytest.mark.usefixtures("offline")


def _results() -> list[grid.ConfigResult]:
    return [
        grid.ConfigResult(
            config_id="dbscan-01",
            method="dbscan",
            params={"eps": 0.5, "min_samples": 3},
            n_clusters=3,
            noise_fraction=0.125,
            silhouette=0.4375,
            largest_cluster_fraction=0.5,
            skipped=None,
        ),
        grid.ConfigResult(
            config_id="kmeans-01",
            method="kmeans",
            params={"k": 60},
            n_clusters=0,
            noise_fraction=1.0,
            silhouette=None,
            largest_cluster_fraction=0.0,
            skipped="needs fewer clusters than skills",
        ),
    ]


def test_grid_roundtrip(tmp_path: Path) -> None:
    labels = np.array([[0, 1, -1, 2], [-2, -2, -2, -2]], dtype=np.int64)
    report.write_grid(tmp_path, _results(), labels)
    lines = (tmp_path / "grid_results.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == ",".join(report.GRID_COLUMNS)
    assert lines[2] == 'kmeans-01,kmeans,"{""k"": 60}",0,1.0,,0.0,needs fewer clusters than skills'
    results, back = report.read_grid(tmp_path)
    assert results == _results()
    assert results[1].silhouette is None and results[0].skipped is None
    assert back.dtype == np.int32 and back.shape == (2, 4)
    assert back.tolist() == labels.tolist()
    with pytest.raises(FileNotFoundError, match="grid_results.csv.*taxonomy grid"):
        report.read_grid(tmp_path / "missing")


def test_context_and_selection_roundtrip(tmp_path: Path) -> None:
    context = {"seed": 0, "n_skills": 7, "nested": {"a": [1, 2]}}
    record = {"config_id": "kmeans-02", "source": "manual", "silhouette": None}
    report.write_context(tmp_path, context)
    report.write_selection(tmp_path, record)
    assert report.read_context(tmp_path) == context
    assert report.read_selection(tmp_path) == record
    assert json.loads((tmp_path / "grid_context.json").read_text()) == context
    assert (tmp_path / "selection.json").exists()
    for read in (report.read_context, report.read_selection):
        with pytest.raises(FileNotFoundError, match="taxonomy grid"):
            read(tmp_path / "nowhere")


def test_review_rows_and_file(tmp_path: Path) -> None:
    skills = ["a", "b", "c", "d", "e", "f"]
    df = [9, 8, 7, 6, 5, 4]
    labels = np.array([1, 0, 1, -1, 0, 1])
    rows = report.review_rows(labels, skills, df, top_members=2)
    assert rows == [
        {"cluster_id": 0, "size": 2, "df_sum": 13, "members": "b; e"},
        {"cluster_id": 1, "size": 3, "df_sum": 20, "members": "a; c"},
        {"cluster_id": -1, "size": 1, "df_sum": 6, "members": "d"},
    ]
    assert all(type(r["size"]) is int and type(r["df_sum"]) is int for r in rows)
    no_noise = report.review_rows(np.array([0, 0, 1]), ["x", "y", "z"], [1, 1, 1])
    assert [r["cluster_id"] for r in no_noise] == [0, 1]
    assert no_noise[0]["members"] == "x; y"
    path = report.write_review(tmp_path, "kmeans-03", rows[:1] + [{**rows[1], "members": "a;b"}])
    assert path == tmp_path / "review_kmeans-03.csv"
    assert path.read_text(encoding="utf-8").splitlines() == [
        "cluster_id,size,df_sum,members",
        "0,2,13,b; e",
        "1,3,20,a;b",
    ]

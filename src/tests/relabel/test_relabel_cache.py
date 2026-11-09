import json
from pathlib import Path

import pytest

from getwork.relabel.cache import RelabelProvenanceError, Stamp, StampedCache

pytestmark = pytest.mark.usefixtures("offline")

STAMP = Stamp(taxonomy_version="sha256:a", model="m1", backend="voyager", prompt_version="p1")


def _filled(path: Path, stamp: Stamp = STAMP) -> None:
    cache = StampedCache(path, stamp)
    cache.put("k1", "Cloud")
    cache.put("k2", [["Rust", "Other"]])
    cache.save()


def test_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "dir" / "c.json"
    cache = StampedCache(path, STAMP)
    assert cache.get("k1") is None and cache.ignored == 0
    _filled(path)
    again = StampedCache(path, STAMP)
    assert again.get("k1") == "Cloud"
    assert again.get("k2") == [["Rust", "Other"]]
    assert again.get("missing") is None
    entry = json.loads(path.read_text())["entries"]["k1"]
    assert entry == {
        "value": "Cloud", "taxonomy_version": "sha256:a", "model": "m1",
        "backend": "voyager", "prompt_version": "p1",
    }
    assert set(json.loads(path.read_text())) == {"entries"}


def test_other_taxonomy_version_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    _filled(path, Stamp("sha256:old", "other-model", "other", "other"))
    cache = StampedCache(path, STAMP)
    assert cache.ignored == 2
    assert cache.get("k1") is None
    cache.save()
    assert json.loads(path.read_text())["entries"] == {}


@pytest.mark.parametrize(
    ("field", "old"), [("model", "m0"), ("backend", "local"), ("prompt_version", "p0")]
)
def test_provenance_mismatch_and_allow_mixed(tmp_path: Path, field: str, old: str) -> None:
    path = tmp_path / "c.json"
    stamp = Stamp(**{**STAMP.__dict__, field: old})
    _filled(path, stamp)
    with pytest.raises(RelabelProvenanceError) as exc:
        StampedCache(path, STAMP)
    message = str(exc.value)
    assert field in message and old in message and getattr(STAMP, field) in message
    assert "k1" not in message and "k2" not in message
    other_fields = {"model", "backend", "prompt_version"} - {field}
    assert not any(f"{f}:" in message for f in other_fields)
    mixed = StampedCache(path, STAMP, allow_mixed=True)
    assert mixed.get("k1") == "Cloud"
    mixed.put("k3", "x")
    mixed.save()
    entries = json.loads(path.read_text())["entries"]
    assert entries["k1"][field] == old
    assert entries["k3"][field] == getattr(STAMP, field)


@pytest.mark.parametrize("content", ["{not json", "[]", '{"entries": []}', '{"entries": {"k": 1}}'])
def test_corrupt_file(tmp_path: Path, content: str) -> None:
    path = tmp_path / "c.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="c.json"):
        StampedCache(path, STAMP)

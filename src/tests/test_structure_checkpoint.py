import json
from pathlib import Path

import pytest

from getwork.structure import store

pytestmark = pytest.mark.usefixtures("offline")


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_read_checkpoint_missing_file_is_empty(tmp_path: Path) -> None:
    assert store.read_checkpoint(tmp_path / "none.jsonl") == {}


def test_read_checkpoint_keeps_valid_unterminated_last_line(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_bytes(b'{"id": "r-1"}\n{"id": "r-2"}')
    assert set(store.read_checkpoint(p)) == {"r-1", "r-2"}
    assert p.read_bytes().endswith(b"\n")
    with store.CheckpointWriter(p) as w:
        w.append({"id": "r-3"})
    assert [r["id"] for r in _lines(p)] == ["r-1", "r-2", "r-3"]


def test_read_checkpoint_drops_unparseable_last_line(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_bytes(b'{"id": "r-1"}\n{"id": "r-2", "x')
    assert set(store.read_checkpoint(p)) == {"r-1"}
    assert p.read_bytes() == b'{"id": "r-1"}\n'


def test_read_checkpoint_no_repair_leaves_file(tmp_path: Path) -> None:
    for content in (b'{"id": "r-1"}\n{"id": "r-2"}', b'{"id": "r-1"}\n{"id": "r'):
        p = tmp_path / "c.jsonl"
        p.write_bytes(content)
        store.read_checkpoint(p, repair=False)
        assert p.read_bytes() == content


@pytest.mark.parametrize("bad", [b"not json\n", b"[1]\n", b'{"id": 3}\n'])
def test_read_checkpoint_bad_complete_line_raises(tmp_path: Path, bad: bytes) -> None:
    p = tmp_path / "c.jsonl"
    p.write_bytes(b'{"id": "r-1"}\n' + bad + b'{"id": "r-2"}\n')
    with pytest.raises(ValueError, match="line 2"):
        store.read_checkpoint(p)


def test_read_checkpoint_first_duplicate_wins(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    p.write_bytes(b'{"id": "r-1", "v": 1}\n{"id": "r-1", "v": 2}\n')
    assert store.read_checkpoint(p)["r-1"]["v"] == 1


def test_checkpoint_writer_appends_and_locks(tmp_path: Path) -> None:
    p = tmp_path / "sub" / "c.jsonl"
    with store.CheckpointWriter(p) as w:
        w.append({"id": "r-1", "t": "é"})
        with pytest.raises(store.AlreadyRunningError):
            store.CheckpointWriter(p)
    assert "é" in p.read_text(encoding="utf-8")
    with store.CheckpointWriter(p) as w:
        w.append({"id": "r-2"})
    assert [r["id"] for r in _lines(p)] == ["r-1", "r-2"]


def test_checkpoint_writer_close_is_idempotent(tmp_path: Path) -> None:
    w = store.CheckpointWriter(tmp_path / "c.jsonl")
    w.close()
    w.close()

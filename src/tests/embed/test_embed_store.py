import json
from pathlib import Path

import numpy as np
import pytest

from getwork.embed.client import EmbeddingMismatchError
from getwork.embed.fake import FakeEmbeddingClient
from getwork.embed.store import check_same_embedding, read_vectors, write_vectors

pytestmark = pytest.mark.usefixtures("offline")


def _write(directory: Path, stem: str = "skills") -> tuple[list[str], np.ndarray, dict]:
    fake = FakeEmbeddingClient(dim=16)
    ids = ["x", "y", "z"]
    vectors = fake.embed(["x ray", "yankee", "zulu"])
    meta = {"embedding": fake.info.as_dict(), "source": "test"}
    write_vectors(directory, stem, ids, vectors, meta)
    return ids, vectors, meta


def test_roundtrip_and_file_names(tmp_path: Path) -> None:
    out = tmp_path / "out"
    ids, vectors, meta = _write(out)
    assert sorted(p.name for p in out.iterdir()) == [
        "skills.npy", "skills_ids.json", "skills_meta.json"
    ]
    loaded = read_vectors(out, "skills")
    assert loaded.ids == ids
    assert loaded.vectors.dtype == np.float32
    assert np.array_equal(loaded.vectors, vectors)
    assert loaded.meta == {**meta, "n": 3, "dim": 16}


def test_write_rejects_bad_vectors_and_meta(tmp_path: Path) -> None:
    fake = FakeEmbeddingClient(dim=16)
    good = fake.embed(["a", "b"])
    meta = {"embedding": fake.info.as_dict()}
    nan = good.copy()
    nan[0, 0] = np.nan
    cases = [
        (["a", "b"], good.astype(np.float64), meta),
        (["a"], good, meta),
        (["a", "a"], good, meta),
        (["a", "b"], nan, meta),
        (["a", "b"], good * 2, meta),
        (["a", "b"], good, {}),
        (["a", "b"], good, {"embedding": {**fake.info.as_dict(), "dim": 8}}),
    ]
    for ids, vectors, bad_meta in cases:
        with pytest.raises(ValueError):
            write_vectors(tmp_path / "o", "s", ids, vectors, bad_meta)


def test_read_errors(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError, match="missing_stem.npy"):
        read_vectors(out, "missing_stem")
    _write(out)
    (out / "skills_ids.json").write_text(json.dumps(["x", "y"]))
    with pytest.raises(ValueError):
        read_vectors(out, "skills")
    (out / "skills_ids.json").unlink()
    with pytest.raises(FileNotFoundError, match="skills_ids.json"):
        read_vectors(out, "skills")


def _meta(model: str = "m", backend: str = "local", dim: int = 4, dtype: str = "float16") -> dict:
    return {"embedding": {"model": model, "backend": backend, "dim": dim, "dtype": dtype}}


def test_check_same_embedding() -> None:
    check_same_embedding(_meta(), {**_meta(dtype="float32"), "other": 1})
    with pytest.raises(EmbeddingMismatchError, match="backend") as err:
        check_same_embedding(_meta(), _meta(backend="voyager"))
    assert "'local'" in str(err.value) and "'voyager'" in str(err.value)
    with pytest.raises(EmbeddingMismatchError) as err:
        check_same_embedding(_meta(), _meta(model="n", dim=8))
    assert "model" in str(err.value) and "dim" in str(err.value)

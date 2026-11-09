import numpy as np
import pytest

from getwork.embed.client import EmbeddingError, EmbeddingInfo, l2_normalize
from getwork.embed.fake import FakeEmbeddingClient

pytestmark = pytest.mark.usefixtures("offline")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def test_info_as_dict_order() -> None:
    info = EmbeddingInfo("m", "local", 8, 2048, "float16")
    assert list(info.as_dict()) == ["model", "backend", "dim", "max_seq_length", "dtype"]
    assert info.as_dict()["dim"] == 8


def test_l2_normalize_rows() -> None:
    data = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)
    original = data.copy()
    out = l2_normalize(data)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(data, original)
    assert out is not data


def test_l2_normalize_rejects_bad_input() -> None:
    for bad in (
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[np.nan, 1.0]]),
        np.array([[np.inf, 1.0]]),
        np.array([1.0, 2.0]),
    ):
        with pytest.raises(EmbeddingError):
            l2_normalize(bad)


def test_fake_is_deterministic_unit_norm() -> None:
    a = FakeEmbeddingClient(dim=64).embed(["python", "Go lang"])
    b = FakeEmbeddingClient(dim=64).embed(["python", "Go lang"])
    assert a.shape == (2, 64)
    assert a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-3)
    assert np.array_equal(a, b)
    assert round(float(a[0][0]), 6) == 0.158828


def test_fake_shared_tokens_raise_cosine() -> None:
    fake = FakeEmbeddingClient()
    base, shared, disjoint = fake.embed(["python django web", "python django api", "kubernetes"])
    assert _cosine(base, shared) > _cosine(base, disjoint)


def test_fake_empty_inputs_and_calls() -> None:
    fake = FakeEmbeddingClient(dim=40)
    assert fake.embed([]).shape == (0, 40)
    blank = fake.embed([""])
    assert blank.shape == (1, 40)
    assert np.isclose(np.linalg.norm(blank[0]), 1.0, atol=1e-3)
    fake.embed(["a", "b"])
    assert fake.calls == [[], [""], ["a", "b"]]
    assert fake.info.backend == "fake"
    assert fake.info.dim == 40

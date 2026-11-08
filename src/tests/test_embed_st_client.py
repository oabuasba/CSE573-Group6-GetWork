import sys
import types
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from getwork.embed import st_client
from getwork.embed.client import EmbeddingError, EmbeddingResourceError
from getwork.embed.fake import FakeEmbeddingClient
from getwork.embed.settings import EmbedSettings

pytestmark = pytest.mark.usefixtures("offline")

DIM = 8


class StandInOutOfMemoryError(RuntimeError):
    pass


class StandInModel:
    instances: list["StandInModel"] = []
    oom_above: int | None = None
    fail_with: Exception | None = None

    def __init__(self, name: str, **kwargs: object) -> None:
        self.name = name
        self.kwargs = kwargs
        self.max_seq_length = 0
        self.batch_sizes: list[int] = []
        self._fake = FakeEmbeddingClient(dim=DIM)
        StandInModel.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int:
        return DIM

    def encode(self, batch: Sequence[str], **kwargs: object) -> np.ndarray:
        self.batch_sizes.append(len(batch))
        if StandInModel.fail_with is not None:
            raise StandInModel.fail_with
        if StandInModel.oom_above is not None and len(batch) > StandInModel.oom_above:
            raise StandInOutOfMemoryError("CUDA failure")
        return self._fake.embed(batch)


@pytest.fixture
def stand_ins(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    StandInModel.instances = []
    StandInModel.oom_above = None
    StandInModel.fail_with = None
    state = types.SimpleNamespace(cuda=True, emptied=0)
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "marker-float16"
    fake_torch.bfloat16 = "marker-bfloat16"
    fake_torch.float32 = "marker-float32"

    def empty_cache() -> None:
        state.emptied += 1

    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: state.cuda,
        OutOfMemoryError=StandInOutOfMemoryError,
        empty_cache=empty_cache,
    )
    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = StandInModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    return state


def make(**kwargs: object) -> st_client.SentenceTransformerClient:
    settings = EmbedSettings(backend="local", model="some/model", **kwargs)
    return st_client.SentenceTransformerClient(settings)


def test_constructor_arguments_and_info(stand_ins: types.SimpleNamespace) -> None:
    client = make(device="cuda", dtype="bfloat16", max_seq_length=512)
    model = StandInModel.instances[0]
    assert model.name == "some/model"
    assert model.kwargs == {
        "device": "cuda",
        "model_kwargs": {"torch_dtype": "marker-bfloat16"},
        "tokenizer_kwargs": {"padding_side": "left"},
    }
    assert model.max_seq_length == 512
    assert client.info.dim == DIM
    assert client.info.backend == "local"
    assert client.info.max_seq_length == 512
    assert client.info.dtype == "bfloat16"


def test_auto_device_and_dtype(stand_ins: types.SimpleNamespace) -> None:
    client = make()
    assert StandInModel.instances[0].kwargs["device"] == "cuda"
    assert StandInModel.instances[0].kwargs["model_kwargs"] == {"torch_dtype": "marker-float16"}
    assert client.info.dtype == "float16"
    stand_ins.cuda = False
    with pytest.raises(EmbeddingError, match="--device cpu"):
        make()
    cpu = make(device="cpu")
    assert StandInModel.instances[-1].kwargs["model_kwargs"] == {"torch_dtype": "marker-float32"}
    assert cpu.info.dtype == "float32"


def test_embed_order_unit_norm_and_empty(stand_ins: types.SimpleNamespace) -> None:
    client = make(batch_size=2)
    texts = ["a", "bb bb", "ccc ccc ccc", "d e", "f"]
    vectors = client.embed(texts)
    expected = FakeEmbeddingClient(dim=DIM).embed(texts)
    assert vectors.shape == (5, DIM) and vectors.dtype == np.float32
    assert np.allclose(vectors, expected, atol=1e-5)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    assert client.embed([]).shape == (0, DIM)


def test_oom_backoff_and_error_wrapping(
    stand_ins: types.SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    client = make(batch_size=8)
    StandInModel.oom_above = 2
    texts = [f"text {i}" for i in range(8)]
    vectors = client.embed(texts)
    assert np.allclose(vectors, FakeEmbeddingClient(dim=DIM).embed(texts), atol=1e-5)
    assert StandInModel.instances[0].batch_sizes[:3] == [8, 4, 2]
    assert stand_ins.emptied == 2
    assert "batch size to 2" in capsys.readouterr().err
    StandInModel.oom_above = 0
    with pytest.raises(EmbeddingResourceError):
        client.embed(["x"])
    StandInModel.oom_above = None
    StandInModel.fail_with = RuntimeError("boom")
    with pytest.raises(EmbeddingError, match="RuntimeError: boom"):
        client.embed(["x"])


def test_missing_library_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbeddingError, match="uv sync --extra embed"):
        make()


def test_no_module_level_heavy_imports() -> None:
    source = Path(st_client.__file__).read_text(encoding="utf-8")
    prefixes = (
        "import torch",
        "from torch",
        "import sentence_transformers",
        "from sentence_transformers",
    )
    offenders = [line for line in source.splitlines() if line.startswith(prefixes)]
    assert offenders == []

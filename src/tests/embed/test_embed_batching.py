from pathlib import Path

import numpy as np
import pytest

from getwork.embed.batching import encode_in_batches
from getwork.embed.cache import embed_cached
from getwork.embed.client import EmbeddingResourceError
from getwork.embed.fake import FakeEmbeddingClient

pytestmark = pytest.mark.usefixtures("offline")

TEXTS = ["a", "bbbb", "cc", "ddddddd", "eee", "ffffffffff", "g", "hh", "iiiii"]


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, MemoryError)


def test_order_restored() -> None:
    fake = FakeEmbeddingClient()
    seen: list[list[str]] = []

    def encode(batch):
        seen.append(list(batch))
        return fake.embed(batch)

    out = encode_in_batches(encode, TEXTS, 4, is_oom=_is_oom)
    assert np.allclose(out, fake.embed(TEXTS))
    assert seen[0][0] == "ffffffffff"
    assert out.dtype == np.float32


def test_oom_halves_batch_size() -> None:
    fake = FakeEmbeddingClient()

    def encode(batch):
        if len(batch) > 2:
            raise MemoryError("out of memory")
        return fake.embed(batch)

    sizes: list[int] = []
    out = encode_in_batches(encode, TEXTS, 8, is_oom=_is_oom, on_oom=sizes.append)
    assert sizes == [4, 2]
    assert np.allclose(out, encode_in_batches(fake.embed, TEXTS, 8, is_oom=_is_oom))


def test_oom_at_size_one_raises() -> None:
    def encode(batch):
        raise MemoryError("out of memory")

    with pytest.raises(EmbeddingResourceError):
        encode_in_batches(encode, TEXTS, 2, is_oom=_is_oom)


def test_other_errors_propagate_and_bad_args() -> None:
    def encode(batch):
        raise KeyError("boom")

    with pytest.raises(KeyError):
        encode_in_batches(encode, TEXTS, 2, is_oom=_is_oom)
    fake = FakeEmbeddingClient()
    with pytest.raises(ValueError):
        encode_in_batches(fake.embed, [], 2, is_oom=_is_oom)
    with pytest.raises(ValueError):
        encode_in_batches(fake.embed, TEXTS, 0, is_oom=_is_oom)


def test_cache_reuses_chunks(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    texts = [f"text {i}" for i in range(5)]
    fake = FakeEmbeddingClient()
    first = embed_cached(fake, texts, cache_dir=cache, chunk_size=2)
    assert len(fake.calls) == 3
    second = embed_cached(fake, texts, cache_dir=cache, chunk_size=2)
    assert len(fake.calls) == 3
    assert np.array_equal(first, second)
    assert first.shape == (5, 64)


def test_cache_invalidated_by_text_or_info(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    texts = [f"text {i}" for i in range(6)]
    fake = FakeEmbeddingClient()
    embed_cached(fake, texts, cache_dir=cache, chunk_size=2)
    stale = set(cache.iterdir())
    changed = [*texts[:3], "other", *texts[4:]]
    embed_cached(fake, changed, cache_dir=cache, chunk_size=2)
    assert len(fake.calls) == 4
    assert stale <= set(cache.iterdir())
    renamed = FakeEmbeddingClient(model="other-model")
    embed_cached(renamed, texts, cache_dir=cache, chunk_size=2)
    assert len(renamed.calls) == 3


def test_cache_disabled_progress_and_empty(tmp_path: Path) -> None:
    fake = FakeEmbeddingClient()
    reports: list[tuple[int, int]] = []
    embed_cached(fake, ["a", "b", "c"], cache_dir=None, chunk_size=2,
                 progress=lambda done, total: reports.append((done, total)))
    embed_cached(fake, ["a", "b", "c"], cache_dir=None, chunk_size=2)
    assert len(fake.calls) == 4
    assert reports == [(1, 2), (2, 2)]
    empty = embed_cached(fake, [], cache_dir=tmp_path / "c")
    assert empty.shape == (0, 64) and empty.dtype == np.float32
    with pytest.raises(ValueError):
        embed_cached(fake, ["a"], cache_dir=None, chunk_size=0)

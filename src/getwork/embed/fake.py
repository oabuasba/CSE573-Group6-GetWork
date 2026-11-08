"""Deterministic offline embedding client for tests."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from hashlib import sha256

import numpy as np

from getwork.embed.client import EmbeddingInfo, l2_normalize

EMPTY_TEXT_TOKEN = "\x00"
BYTE_CENTER = 127.5
DIGEST_BYTES = 32


class FakeEmbeddingClient:
    def __init__(self, dim: int = 64, model: str = "fake-embedding") -> None:
        self._info = EmbeddingInfo(model, "fake", dim)
        self.calls: list[list[str]] = []

    @property
    def info(self) -> EmbeddingInfo:
        return self._info

    def _token_vector(self, token: str) -> np.ndarray:
        dim = self._info.dim
        raw = b"".join(
            sha256(f"{i}:{token}".encode()).digest() for i in range(math.ceil(dim / DIGEST_BYTES))
        )[:dim]
        return np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - np.float32(BYTE_CENTER)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls.append(list(texts))
        dim = self._info.dim
        if not texts:
            return np.zeros((0, dim), np.float32)
        rows = np.zeros((len(texts), dim), np.float32)
        for i, text in enumerate(texts):
            tokens = re.findall(r"\w+", text.casefold()) or [EMPTY_TEXT_TOKEN]
            for token in tokens:
                rows[i] += self._token_vector(token)
        return l2_normalize(rows)

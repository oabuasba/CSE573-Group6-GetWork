"""Local Qwen3-Embedding client over sentence-transformers (torch loads only in __init__)."""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np

from getwork.embed import batching
from getwork.embed.client import EmbeddingError, EmbeddingInfo, l2_normalize
from getwork.embed.settings import EmbedSettings

MISSING_LIBRARY_MESSAGE = (
    "sentence-transformers and torch are not installed; run: uv sync --extra embed"
)
NO_CUDA_MESSAGE = (
    "no CUDA device found; pass --device cpu to run on the CPU (very slow for a 4B model)"
)


class SentenceTransformerClient:
    def __init__(self, settings: EmbedSettings) -> None:
        try:
            import torch
            import sentence_transformers
        except ImportError:
            raise EmbeddingError(MISSING_LIBRARY_MESSAGE) from None
        self._torch = torch
        self._settings = settings
        self._device = self._pick_device(settings.device)
        dtype_name = settings.dtype
        if dtype_name == "auto":
            dtype_name = "float16" if self._device.startswith("cuda") else "float32"
        self._model = sentence_transformers.SentenceTransformer(
            settings.model,
            device=self._device,
            model_kwargs={"torch_dtype": getattr(torch, dtype_name)},
            tokenizer_kwargs={"padding_side": "left"},
        )
        self._model.max_seq_length = settings.max_seq_length
        self._info = EmbeddingInfo(
            settings.model,
            "local",
            int(self._model.get_sentence_embedding_dimension()),
            settings.max_seq_length,
            dtype_name,
        )

    def _pick_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        if not self._torch.cuda.is_available():
            raise EmbeddingError(NO_CUDA_MESSAGE)
        return "cuda"

    @property
    def info(self) -> EmbeddingInfo:
        return self._info

    def _encode(self, batch: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(batch),
            batch_size=len(batch),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _is_oom(self, exc: BaseException) -> bool:
        return isinstance(exc, self._torch.cuda.OutOfMemoryError) or (
            "out of memory" in str(exc).lower()
        )

    def _on_oom(self, new_batch_size: int) -> None:
        if self._device.startswith("cuda"):
            self._torch.cuda.empty_cache()
        print(f"out of memory: reducing batch size to {new_batch_size}", file=sys.stderr)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self._info.dim), dtype=np.float32)
        try:
            rows = batching.encode_in_batches(
                self._encode,
                texts,
                self._settings.batch_size,
                is_oom=self._is_oom,
                on_oom=self._on_oom,
            )
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"{type(e).__name__}: {e}") from None
        return l2_normalize(rows)

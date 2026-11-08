"""embed-preflight: embed three probe words with the chosen backend and print a JSON report."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping

import numpy as np

from getwork.embed import factory
from getwork.embed.client import EmbeddingClient, EmbeddingError
from getwork.llm.client import LLMError, redact_secrets
from getwork.llm.settings import ConfigError

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_EMBED_FAILED = 3
PROBE_WORDS = ["Python", "Django", "Kubernetes"]
NORM_TOLERANCE = 1e-3


def _build_report(client: EmbeddingClient, *, seconds: float, vectors: np.ndarray) -> dict:
    info = client.info
    norms = np.linalg.norm(vectors, axis=1)
    return {
        "backend": info.backend,
        "model": info.model,
        "dim": info.dim,
        "max_seq_length": info.max_seq_length,
        "dtype": info.dtype,
        "seconds": round(seconds, 3),
        "related_cosine": float(vectors[0] @ vectors[1]),
        "unrelated_cosine": float(vectors[0] @ vectors[2]),
        "norms_ok": bool(np.all(np.abs(norms - 1.0) <= NORM_TOLERANCE)),
    }


def main(
    argv: list[str] | None = None,
    *,
    client: EmbeddingClient | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="embed-preflight", description=__doc__)
    factory.add_embed_args(parser)
    args = parser.parse_args(argv)
    try:
        bundle = factory.make_embedding_client(
            args=args, environ=os.environ if environ is None else environ, client=client
        )
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        started = time.perf_counter()
        vectors = bundle.client.embed(PROBE_WORDS)
        seconds = time.perf_counter() - started
    except (EmbeddingError, LLMError) as exc:
        print(f"embedding failed: {redact_secrets(str(exc), bundle.secrets)}", file=sys.stderr)
        return EXIT_EMBED_FAILED
    report = _build_report(bundle.client, seconds=seconds, vectors=vectors)
    print(json.dumps(report, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

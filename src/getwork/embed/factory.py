"""Backend selection for embedding clients; heavy libraries load only in _real_client."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from getwork.embed import settings as embed_settings
from getwork.embed.client import EMBEDDING_BACKENDS, EmbeddingClient, EmbeddingInfo


@dataclass(frozen=True)
class EmbeddingBundle:
    client: EmbeddingClient
    info: EmbeddingInfo
    secrets: list[str]


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def add_embed_args(parser: argparse.ArgumentParser, *, with_env_file: bool = True) -> None:
    parser.add_argument("--embed-backend", choices=EMBEDDING_BACKENDS, default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--embed-batch-size", type=_positive_int, default=None)
    parser.add_argument("--max-seq-length", type=_positive_int, default=None)
    if with_env_file:
        parser.add_argument("--env-file", default=".env")


def make_embedding_client(
    *,
    args: argparse.Namespace,
    environ: Mapping[str, str],
    client: EmbeddingClient | None = None,
) -> EmbeddingBundle:
    if client is not None:
        return EmbeddingBundle(client=client, info=client.info, secrets=[])
    settings = embed_settings.load_embed_settings(
        environ,
        Path(args.env_file),
        backend=args.embed_backend,
        model=args.embed_model,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.embed_batch_size,
        max_seq_length=args.max_seq_length,
    )
    real = _real_client(settings)
    secrets = [settings.api_key] if settings.backend == "voyager" else []
    return EmbeddingBundle(client=real, info=real.info, secrets=secrets)


def _real_client(settings: embed_settings.EmbedSettings) -> EmbeddingClient:
    if settings.backend == "voyager":
        from getwork.embed.voyager_embed import VoyagerEmbeddingClient

        return VoyagerEmbeddingClient(settings)
    from getwork.embed.st_client import SentenceTransformerClient

    return SentenceTransformerClient(settings)

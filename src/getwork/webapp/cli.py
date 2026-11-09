"""The webapp command: serve and check (demo mode only until the artifact engine exists)."""

import argparse
import logging
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from getwork.embed import factory as embed_factory
from getwork.embed.client import EmbeddingClient
from getwork.llm import client as llm_client_mod
from getwork.llm import factory as llm_factory
from getwork.llm.client import LLMClient
from getwork.retrieval import b1plus
from getwork.webapp import api, demo, engine, explain, models

WEB_DIST: Path = Path(__file__).resolve().parents[2] / "web" / "dist"
DEFAULT_ARTIFACTS_DIR = "artifacts_trial"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
ALL_SYSTEMS: tuple[str, ...] = ("B1+", "B2", "Full")
NOT_BUILT_MESSAGE = "artifact mode is not built yet; use --demo"
BUILD_COMMAND = "npm --prefix src/web run build"
STRUCTURE_DISABLED = "Disabled with --no-structure."
EMBED_DISABLED = "Disabled with --no-embed."
LOG_LEVEL = "info"


class UsageError(ValueError):
    """A bad option value; the message is printed to stderr and the exit code is 2."""


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    common.add_argument("--demo", action="store_true", help="use built-in demo data")
    common.add_argument("--systems", default="all", help="all or a list of B1+,B2,Full")
    common.add_argument("--b1plus-weights", default=None, help="JSON file for load_weights")
    common.add_argument("--max-query-chars", type=_positive_int, default=20000)
    common.add_argument("--no-structure", action="store_true")
    common.add_argument("--no-embed", action="store_true")
    llm_factory.add_llm_args(common)
    embed_factory.add_embed_args(common, with_env_file=False)
    parser = argparse.ArgumentParser(prog="webapp", description="GetWork web app")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", parents=[common], help="start the server")
    serve.add_argument("--host", default=DEFAULT_HOST)
    serve.add_argument("--port", type=_positive_int, default=DEFAULT_PORT)
    serve.add_argument("--static-dir", default=None)
    commands.add_parser("check", parents=[common], help="load everything and print health")
    return parser


def _check_options(args: argparse.Namespace) -> None:
    if args.systems != "all":
        unknown = [s for s in args.systems.split(",") if s not in ALL_SYSTEMS]
        if unknown or not args.systems:
            raise UsageError(f"unknown --systems value {unknown or args.systems!r}")
    if args.b1plus_weights is not None:
        try:
            b1plus.load_weights(Path(args.b1plus_weights))
        except (OSError, ValueError) as exc:
            raise UsageError(f"--b1plus-weights: {exc}") from exc


class _CapabilityOverride:
    """Wraps an engine so --no-structure and --no-embed show up in health."""

    def __init__(self, inner: engine.Engine, *, no_structure: bool, no_embed: bool) -> None:
        self._inner = inner
        self._disabled = {
            "structure": STRUCTURE_DISABLED if no_structure else None,
            "embed": EMBED_DISABLED if no_embed else None,
        }

    def health(self) -> models.EngineHealth:
        health = self._inner.health()
        capabilities = dict(health.capabilities)
        for name, reason in self._disabled.items():
            if reason:
                capabilities[name] = models.Capability(available=False, reason=reason)
        return health.model_copy(update={"capabilities": capabilities})

    def examples(self, side: models.Side, limit: int) -> list[models.DocumentView]:
        return self._inner.examples(side, limit)

    def get_document(self, doc_id: str) -> models.DocumentView:
        return self._inner.get_document(doc_id)

    def match(self, request: models.MatchRequest) -> engine.MatchResult:
        return self._inner.match(request)


def _build_engine(args: argparse.Namespace) -> engine.Engine:
    if not args.demo:
        raise UsageError(NOT_BUILT_MESSAGE)
    return _CapabilityOverride(
        demo.DemoEngine(), no_structure=args.no_structure, no_embed=args.no_embed
    )


class RedactingFilter(logging.Filter):
    """Redacts secrets from a record's message and traceback before any handler formats it."""

    def __init__(self, secrets: Sequence[str]) -> None:
        super().__init__()
        self._secrets = list(secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = llm_client_mod.redact_secrets(record.getMessage(), self._secrets)
        record.args = ()
        if record.exc_info:
            text = "".join(traceback.format_exception(*record.exc_info))
            record.exc_text = llm_client_mod.redact_secrets(text, self._secrets)
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = llm_client_mod.redact_secrets(record.exc_text, self._secrets)
        return True


def install_uvicorn_redaction(secrets: Sequence[str]) -> None:
    """Uvicorn logs the traceback of unhandled errors itself, so redact on its logger too."""
    logger = logging.getLogger("uvicorn.error")
    for existing in [f for f in logger.filters if isinstance(f, RedactingFilter)]:
        logger.removeFilter(existing)
    logger.addFilter(RedactingFilter(secrets))


def _static_dir(args: argparse.Namespace) -> Path | None:
    if args.static_dir is not None:
        return Path(args.static_dir)
    return WEB_DIST if (WEB_DIST / "index.html").is_file() else None


def _serve(
    args: argparse.Namespace, app_engine: engine.Engine, run: Callable[..., None] | None
) -> int:
    static_dir = _static_dir(args)
    app = api.create_app(app_engine, explain.TemplateExplainer(), static_dir=static_dir, secrets=())
    if args.host not in LOOPBACK_HOSTS:
        print(
            f"warning: {args.host} is a public address and the server has no authentication; "
            "anyone who can reach it can use it.",
            file=sys.stderr,
        )
    print(f"GetWork is serving on http://{args.host}:{args.port} (API docs at /docs)")
    if static_dir is None:
        print(f"The web UI is not built yet; build it with: {BUILD_COMMAND}")
    install_uvicorn_redaction(())
    if run is None:
        import uvicorn

        run = uvicorn.run
    run(app, host=args.host, port=args.port, log_level=LOG_LEVEL)
    return 0


def _check(app_engine: engine.Engine) -> int:
    health = models.HealthResponse(
        **app_engine.health().model_dump(), explainer=explain.TemplateExplainer.info
    )
    print(health.model_dump_json(indent=2))
    return 0


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    llm_client: LLMClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    run: Callable[..., None] | None = None,
) -> int:
    """Exit codes: 0 ok, 2 usage, configuration or artifact error. Never prints a key.

    environ, llm_client and embedding_client are used by artifact mode, which is not built yet.
    """
    del environ, llm_client, embedding_client
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exit_request:
        return int(exit_request.code or 0)
    try:
        _check_options(args)
        app_engine = _build_engine(args)
    except (UsageError, engine.EngineLoadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.command == "check":
        return _check(app_engine)
    return _serve(args, app_engine, run)

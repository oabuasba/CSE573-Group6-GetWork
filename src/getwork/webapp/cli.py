"""The webapp command: serve and check, in demo mode or over an artifacts run folder."""

import argparse
import logging
import os
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from getwork.embed import factory as embed_factory
from getwork.embed.client import EmbeddingClient
from getwork.llm import client as llm_client_mod
from getwork.llm import factory as llm_factory
from getwork.llm.client import LLMClient
from getwork.llm.settings import BACKENDS, ConfigError
from getwork.retrieval import b1plus
from getwork.structure.extract import ExtractConfig
from getwork.webapp import api, artifacts, demo, engine, explain, models

if TYPE_CHECKING:
    from getwork.agent.explainer import AgentExplainer, ChatService

AgentParts = tuple["AgentExplainer", "ChatService", Sequence[str]]

WEB_DIST: Path = Path(__file__).resolve().parents[2] / "web" / "dist"
DEFAULT_ARTIFACTS_DIR = "artifacts_trial"
ARTIFACTS_DIR_ENV = "WEBAPP_ARTIFACTS_DIR"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
ALL_SYSTEMS: tuple[str, ...] = ("B1+", "B2", "Full")
BUILD_COMMAND = "npm --prefix src/web run build"
STRUCTURE_DISABLED = "Disabled with --no-structure."
EMBED_DISABLED = "Disabled with --no-embed."
LOG_LEVEL = "info"
DEFAULT_AGENT_TIMEOUT_S = 60.0
EXPLAINERS = ("template", "agent")


class UsageError(ValueError):
    """A bad option value; the message is printed to stderr and the exit code is 2."""


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return value


def _build_parser(environ: Mapping[str, str] | None = None) -> argparse.ArgumentParser:
    """--artifacts-dir wins over WEBAPP_ARTIFACTS_DIR, which wins over the artifacts_trial default."""
    env = os.environ if environ is None else environ
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--artifacts-dir",
        default=env.get(ARTIFACTS_DIR_ENV) or DEFAULT_ARTIFACTS_DIR,
        help=f"a run folder or a folder of vN runs (env {ARTIFACTS_DIR_ENV}; default "
        f"{DEFAULT_ARTIFACTS_DIR})",
    )
    common.add_argument("--demo", action="store_true", help="use built-in demo data")
    common.add_argument("--systems", default="all", help="all or a list of B1+,B2,Full")
    common.add_argument("--b1plus-weights", default=None, help="JSON file for load_weights")
    common.add_argument("--max-query-chars", type=_positive_int, default=20000)
    common.add_argument("--no-structure", action="store_true")
    common.add_argument("--no-embed", action="store_true")
    llm_factory.add_llm_args(common)
    common.add_argument("--explainer", choices=EXPLAINERS, default="template")
    common.add_argument("--agent-backend", choices=BACKENDS, default=None)
    common.add_argument("--agent-model", default=None)
    common.add_argument("--agent-timeout", type=_positive_float, default=DEFAULT_AGENT_TIMEOUT_S)
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


def _structure_resources(
    args: argparse.Namespace, environ: Mapping[str, str], llm_client: LLMClient | None
) -> tuple[artifacts.StructureResources | None, str | None]:
    if args.no_structure:
        return None, STRUCTURE_DISABLED
    try:
        bundle = llm_factory.make_client(
            backend=args.backend, model=args.model, env_file=Path(args.env_file),
            environ=environ, client=llm_client,
        )
        cfg = ExtractConfig(
            model=bundle.model, backend=bundle.backend,
            extra_body=llm_factory.parse_extra_body(args.extra_body),
        )
    except ConfigError as exc:  # names variables, never values
        return None, str(exc)
    return artifacts.StructureResources(client=bundle.client, cfg=cfg, secrets=bundle.secrets), None


def _embed_resources(
    args: argparse.Namespace, environ: Mapping[str, str], embedding_client: EmbeddingClient | None
) -> tuple[artifacts.EmbedResources | None, str | None]:
    if args.no_embed:
        return None, EMBED_DISABLED
    if embedding_client is not None:
        info = embedding_client.info
        describe = f"{info.backend}, {info.model}"
    else:
        named = [part for part in (args.embed_backend, args.embed_model) if part]
        describe = ", ".join(named) or "configured in .env"

    def factory() -> EmbeddingClient:
        return embed_factory.make_embedding_client(
            args=args, environ=environ, client=embedding_client
        ).client

    return artifacts.EmbedResources(factory=factory, describe=describe), None


def _build_engine(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str],
    llm_client: LLMClient | None,
    embedding_client: EmbeddingClient | None,
) -> engine.Engine:
    if args.demo:
        return _CapabilityOverride(
            demo.DemoEngine(), no_structure=args.no_structure, no_embed=args.no_embed
        )
    structure, structure_reason = _structure_resources(args, environ, llm_client)
    embed, embed_reason = _embed_resources(args, environ, embedding_client)
    systems = ALL_SYSTEMS if args.systems == "all" else tuple(args.systems.split(","))
    weights = (
        b1plus.load_weights(Path(args.b1plus_weights)) if args.b1plus_weights
        else b1plus.DEFAULT_WEIGHTS
    )
    return artifacts.ArtifactEngine.load(
        Path(args.artifacts_dir), systems=systems, weights=weights, structure=structure,
        structure_reason=structure_reason, embed=embed, embed_reason=embed_reason,
        max_query_chars=args.max_query_chars,
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


def build_agent(
    args: argparse.Namespace,
    *,
    engine: engine.Engine,
    environ: Mapping[str, str],
    llm_client: LLMClient | None,
) -> AgentParts:
    """The agent explainer and chat service over one client (ADR-21: the structuring backend and model unless
    --agent-backend or --agent-model say otherwise), and the secrets to redact. Imports the agent lazily."""
    from getwork.agent import explainer, graph, state, turn, vocab

    bundle = llm_factory.make_client(
        backend=args.agent_backend or args.backend,
        model=args.agent_model or args.model,
        env_file=Path(args.env_file),
        environ=environ,
        client=llm_client,
        timeout_s=args.agent_timeout,
    )
    cfg = state.AgentConfig(
        model=bundle.model,
        backend=bundle.backend,
        timeout_s=args.agent_timeout,
        extra_body=llm_factory.parse_extra_body(args.extra_body),
    )
    vocabulary = (
        vocab.demo_vocabulary() if args.demo else vocab.load_vocabulary(_loaded_root(args, engine))
    )
    runner = turn.AgentRunner(client=bundle.client, cfg=cfg, vocab=vocabulary, secrets=bundle.secrets)
    compiled = graph.build_graph(runner)
    return (
        explainer.AgentExplainer(graph=compiled, runner=runner, engine=engine),
        explainer.ChatService(graph=compiled, runner=runner, engine=engine),
        bundle.secrets,
    )


def _loaded_root(args: argparse.Namespace, loaded: engine.Engine) -> Path:
    """The run folder the engine actually resolved (e.g. artifacts_trial/v5), not the raw option."""
    resolved = loaded.health().artifacts_dir
    return Path(resolved if resolved else args.artifacts_dir)


def _static_dir(args: argparse.Namespace) -> Path | None:
    if args.static_dir is not None:
        return Path(args.static_dir)
    return WEB_DIST if (WEB_DIST / "index.html").is_file() else None


def _serve(
    args: argparse.Namespace,
    app_engine: engine.Engine,
    run: Callable[..., None] | None,
    agent: AgentParts | None,
) -> int:
    static_dir = _static_dir(args)
    secrets: Sequence[str] = agent[2] if agent else ()
    explainer = agent[0] if agent else explain.TemplateExplainer()
    app = api.create_app(app_engine, explainer, static_dir=static_dir, secrets=secrets)
    if agent:
        from getwork.agent.api import register_chat_routes

        register_chat_routes(app, agent[1], secrets=secrets)
    if args.host not in LOOPBACK_HOSTS:
        print(
            f"warning: {args.host} is a public address and the server has no authentication; "
            "anyone who can reach it can use it.",
            file=sys.stderr,
        )
    print(f"GetWork is serving on http://{args.host}:{args.port} (API docs at /docs)")
    print(f"Artifacts: {app_engine.health().artifacts_dir or 'demo data'}")
    if static_dir is None:
        print(f"The web UI is not built yet; build it with: {BUILD_COMMAND}")
    install_uvicorn_redaction(secrets)
    if run is None:
        import uvicorn

        run = uvicorn.run
    run(app, host=args.host, port=args.port, log_level=LOG_LEVEL)
    return 0


def _check(app_engine: engine.Engine, info: models.ExplainerInfo) -> int:
    health = models.HealthResponse(**app_engine.health().model_dump(), explainer=info)
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

    environ and the injected clients replace os.environ and the real LLM and embedding clients (tests).
    """
    env = os.environ if environ is None else environ
    try:
        args = _build_parser(env).parse_args(argv)
    except SystemExit as exit_request:
        return int(exit_request.code or 0)
    try:
        _check_options(args)
        app_engine = _build_engine(
            args, environ=env, llm_client=llm_client, embedding_client=embedding_client
        )
    except (UsageError, engine.EngineLoadError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    agent = None
    if args.explainer == "agent":
        try:
            agent = build_agent(
                args,
                engine=app_engine,
                environ=env,
                llm_client=llm_client,
            )
        except (ConfigError, RuntimeError, ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.command == "check":
        return _check(app_engine, agent[0].info if agent else explain.TemplateExplainer.info)
    return _serve(args, app_engine, run, agent)

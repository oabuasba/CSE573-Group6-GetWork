"""Command line entry point for the structurer (parser, config errors, dry-run)."""

import argparse
import json
import os
import signal
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

from getwork.llm import settings as llm_settings
from getwork.llm.client import LLMClient, LLMError, describe_exception
from getwork.structure import extract, inputs, prompt, runner, runstate, store

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ABORTED = runstate.EXIT_ABORTED
EXIT_INTERRUPTED = runstate.EXIT_INTERRUPTED
EXAMPLE_MESSAGE_CHARS = 500
REASONING_CHOICES = ["none", "low", "medium", "high", "omit"]


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {value}")
    return number


def _temperature(value: str) -> float | None:
    if value == "omit":
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number or 'omit': {value!r}") from None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="structurer", description="Extract structured records from cleaned resumes/postings."
    )
    p.add_argument("--source", required=True, choices=["resumes", "postings"])
    p.add_argument("--in-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--prompt-version", default=prompt.PROMPT_VERSION)
    p.add_argument("--workers", type=_positive_int, default=4)
    p.add_argument("--limit", type=_positive_int, default=None)
    p.add_argument("--sample", type=_positive_int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=_positive_int, default=4096)
    p.add_argument("--max-input-chars", type=_positive_int, default=48000)
    p.add_argument("--temperature", type=_temperature, default=0.0)
    p.add_argument("--reasoning-effort", choices=REASONING_CHOICES, default="none")
    p.add_argument("--no-response-schema", action="store_true")
    p.add_argument("--max-consecutive-failures", type=int, default=25)
    p.add_argument("--allow-mixed", action="store_true")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--preflight", action="store_true")
    return p


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.sample is not None:
        parser.error("--limit and --sample are mutually exclusive")
    if args.in_dir is None:
        args.in_dir = Path("data/processed") / args.source
    if args.out_dir is None:
        args.out_dir = Path("data/structured") / args.source
    return args


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_USAGE


def _dry_run(args: argparse.Namespace) -> int:
    df = inputs.load_inputs(source=args.source, in_dir=args.in_dir)
    paths = store.output_paths(source=args.source, out_dir=args.out_dir)
    done = store.read_checkpoint(paths.jsonl, repair=False)
    universe = inputs.select_universe(
        all_ids=list(df["id"]), limit=args.limit, sample=args.sample, seed=args.seed
    )
    todo = [i for i in universe if i not in done]
    example = None
    if todo:
        row = df[df["id"] == todo[0]].iloc[0]
        title = row["title"] if "title" in df.columns else None
        example = prompt.build_user_message(
            source=args.source, title=title, text=row["text_clean"]
        )[:EXAMPLE_MESSAGE_CHARS]
    report = {
        "source": args.source,
        "in_dir": str(args.in_dir),
        "out_dir": str(args.out_dir),
        "total_in_source": len(df),
        "selected": len(universe),
        "already_done": len(universe) - len(todo),
        "to_process": len(todo),
        "example_user_message": example,
    }
    print(json.dumps(report))
    return EXIT_OK


def _build_real_client(
    args: argparse.Namespace, environ: Mapping[str, str]
) -> tuple[LLMClient, str, list[str]]:
    settings = llm_settings.load_settings(
        environ=environ, dotenv_path=Path(args.env_file), model_override=args.model
    )
    from getwork.llm import voyager  # lazy: keeps openai out of module import (AC-30)

    return voyager.build_client(settings), settings.model, [settings.api_key]


def _preflight(client: LLMClient, model: str, secrets: list[str]) -> int:
    describe = getattr(client, "describe_model", None)
    if describe is None:
        return _error("this client does not support --preflight")
    try:
        info = describe(model)
    except LLMError as exc:
        print(f"error: preflight failed: {describe_exception(exc, secrets)}", file=sys.stderr)
        return EXIT_ABORTED
    print(json.dumps(info))
    return EXIT_OK if info.get("found") else EXIT_ABORTED


def _build_run_config(args: argparse.Namespace, model: str) -> runstate.RunConfig:
    extract_cfg = extract.ExtractConfig(
        model=model,
        prompt_version=args.prompt_version,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=None if args.reasoning_effort == "omit" else args.reasoning_effort,
        use_response_schema=not args.no_response_schema,
        max_input_chars=args.max_input_chars,
    )
    return runstate.RunConfig(
        source=args.source,
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        extract=extract_cfg,
        workers=args.workers,
        limit=args.limit,
        sample=args.sample,
        seed=args.seed,
        max_consecutive_failures=args.max_consecutive_failures,
        allow_mixed=args.allow_mixed,
    )


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _execute(args: argparse.Namespace, client: LLMClient, model: str, secrets: list[str]) -> int:
    if args.preflight:
        return _preflight(client=client, model=model, secrets=secrets)
    cfg = _build_run_config(args=args, model=model)
    stop = threading.Event()
    previous = None
    is_main = threading.current_thread() is threading.main_thread()
    if is_main:

        def on_sigint(signum: int, frame: object) -> None:
            stop.set()
            signal.signal(signal.SIGINT, signal.SIG_DFL)

        previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        result = runner.run(cfg, client, secrets=secrets, stop=stop, log=_log)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    finally:
        if is_main:
            signal.signal(signal.SIGINT, previous)
    print(json.dumps(result.summary))
    return result.exit_code


def _run(
    args: argparse.Namespace, client: LLMClient | None, environ: Mapping[str, str] | None
) -> int:
    try:
        prompt.load_system_prompt(args.prompt_version)
    except FileNotFoundError:
        return _error(f"unknown prompt version: {args.prompt_version}")
    if args.dry_run:
        return _dry_run(args)
    secrets: list[str] = []
    if client is None:
        client, model, secrets = _build_real_client(
            args=args, environ=os.environ if environ is None else environ
        )
    elif not args.model:
        return _error("--model is required")
    else:
        model = args.model
    return _execute(args=args, client=client, model=model, secrets=secrets)


def main(
    argv: list[str] | None = None,
    *,
    client: LLMClient | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Exit codes: 0 ok, 1 finished with record failures, 2 usage/config error, 3 aborted or
    preflight failed, 130 interrupted. argparse usage errors raise SystemExit(2)."""
    args = _parse_args(argv)
    try:
        return _run(args=args, client=client, environ=environ)
    except llm_settings.ConfigError as exc:
        return _error(str(exc))
    except (
        FileNotFoundError,
        ValueError,
        store.AlreadyRunningError,
        runner.ProvenanceMismatchError,
    ) as exc:
        return _error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

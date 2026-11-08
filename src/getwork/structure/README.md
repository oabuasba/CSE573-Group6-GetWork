# Structurer runbook

`structurer` reads the cleaned resumes and postings from `data/processed/<source>/`, sends each
document to an LLM on ASU's Voyager service, and writes one structured record per document
(title, skills, experience, education, and so on). Records are checkpointed as they finish, so an
interrupted run resumes where it stopped. Linux or WSL only.

## Configure

Copy `.env.example` to `.env` (git ignores it) and set three variables:

- `VOYAGER_API_KEY`: created in the Voyager portal (shown once). Never commit it.
- `VOYAGER_BASE_URL`: must include `/v1`, for example `https://openai.rc.asu.edu/v1`.
- `VOYAGER_MODEL`: a model id from `GET /v1/models` (see `--preflight`).

## Commands

```
uv run structurer --source resumes --preflight
uv run structurer --source resumes --dry-run
uv run structurer --source resumes --limit 5 --out-dir data/structured_trial/resumes
uv run structurer --source resumes --sample 100 --seed 0 --out-dir data/structured_trial/resumes
uv run structurer --source resumes            # full run into data/structured/resumes
uv run structurer --source resumes            # resuming: run the same command again
uv run structurer --source resumes --prompt-version extract_v2 --out-dir data/structured_trial/resumes
```

Use `--source postings` for postings. Trial and sample runs always use
`--out-dir data/structured_trial/<source>` (git-ignored) so they never touch the committed
directory.

## Exit codes

- 0: complete.
- 1: complete, but some records failed (they are retried on the next run).
- 2: usage or configuration error.
- 3: run aborted or preflight failed.
- 130: interrupted.

## Output files

In `<out-dir>` (default `data/structured/<source>/`):

- `<source>_structured.jsonl`: the checkpoint, one record per line, appended as records finish.
- `<source>_structured.parquet`: the final sorted output.
- `<source>_structured_failures.csv`: records that failed (stage, error type, message, attempts).
- `<source>_structured_summary.json`: counts (attempted, succeeded, failed, retries, rate_limited), model, prompt_version, status, abort_reason.

## Stopping and run time

Stopping can take time. The first Ctrl-C waits for in-flight calls (up to the 30-minute client
timeout). A second Ctrl-C exits at once, and those records are redone next run. A full run
(37,265 calls) may take a day or more, so keep the machine awake.

## User-run real-API checklist

Run once before a full run. No automated test performs these steps.

### U-1 Create the key and configure .env

Create an API key in the Voyager portal (voyager.rc.asu.edu, ASURITE login, ASU network or VPN
to reach the portal; the key is shown once). Copy `.env.example` to `.env`, fill in the three
variables, and run `git status`.

Pass: `.env` does not appear in `git status`. The base URL ends in `/v1`.

### U-2 Sync the environment

Run `uv sync`.

Pass: it succeeds, and `uv run structurer --help` lists the options.

### U-3 Preflight

Run `uv run structurer --source resumes --preflight`.

Pass: it prints `found: true` for your model. `supports_response_schema` is true (if not, add
`--no-response-schema` to every run). `context_window` is at least
(max_input_chars / 3 + max_tokens + 2,000) tokens, about 22,000 with the defaults (else lower
`--max-input-chars`). No status of deprecated or retired, and no replacement. A 404 with
`{"detail":"Not Found"}` means a legacy key: rotate it in the portal. If the model rejects a
parameter, use `--reasoning-effort omit` or `--temperature omit`.

### U-4 Dry run

Run `uv run structurer --source resumes --dry-run` and the same for postings.

Pass: `total_in_source` is 28,842 and 8,423. `example_user_message` looks right.

### U-5 Five-record trial

Run `uv run structurer --source resumes --limit 5 --out-dir data/structured_trial/resumes` and
the same for postings. Open the JSONL files and compare each record with the text in
`data/processed/<source>/<source>_clean_preview.json` (or the parquet).

Pass: the summary shows `succeeded: 5`, `failed: 0`, `retries: 0`. Titles are plausible, skills
are the document's own wording, postings list requirements (not benefits), durations are
plausible, and nothing is invented.

### U-6 Sample of 100

Run `--sample 100 --seed 0 --out-dir data/structured_trial/<source>` for each source. Delete the
trial directory first if U-5 used a different model or prompt version. To try a revised prompt,
add `prompts/extract_v2.md`, run with `--prompt-version extract_v2` into a fresh trial
directory, and, once accepted, change the `PROMPT_VERSION` default in `prompt.py`.

Pass: failures are under 2% of attempted. For validation failures, read the messages in the
failures CSV: an empty reply means raise `--max-tokens` or check that reasoning is off; anything
else means revise the prompt as a new version (do not edit v1). Time the run with the wall clock
(for example `time uv run structurer ...`) and note `rate_limited` to project a full run (37,265
calls, which can take a day or more). Estimate the full output size (trial JSONL size x N/100);
each committed file must stay under 100 MB.

### U-7 Full runs

Run `uv run structurer --source resumes`, then `uv run structurer --source postings`, into the
default directory. Keep the machine awake. If interrupted (Ctrl-C once, a crash, a lost
connection), run the same command again.

Pass: the second run reports `already_done` equal to the earlier `done_total`. If many 429s
occur, lower `--workers` (try 2). Re-run until `failures_total` stops changing. Exit code 1 means
some records failed and are retried on the next run. One Ctrl-C waits for in-flight calls (up to
the 30-minute client timeout); a second Ctrl-C exits at once and those records are redone next
run.

### U-8 Review and commit

Review and commit `data/structured/` once, after the final run. Record the `model` value from
the summary in the project notes.

Pass: both summaries show one model id and `failed: 0`, or the remaining failures are understood
and listed.

## Local LLM backend

The structurer can send its requests to a model served on your own machine instead of Voyager.
The same client is used with a different configuration (`--backend local`), so any server that
speaks the OpenAI chat-completions dialect and `/v1/models` works (vLLM, Ollama, the llama.cpp
server). Every server step below is run by you; no automated test starts a server, downloads
weights, or measures speed. Items marked inferred come from general knowledge of the tools, not
from a source read for this runbook. Check them against the tool's current documentation first.

Settings (in `.env`, see `.env.example`): `LLM_BACKEND=local`, `LOCAL_LLM_BASE_URL`,
`LOCAL_LLM_MODEL`, and optionally `LOCAL_LLM_API_KEY` (only if the server was started with a
key). `--backend local` on the command line beats `LLM_BACKEND`. Voyager settings in the same
file are ignored while the backend is local.

Behavior that differs from Voyager:

- Defaults for local: 2 workers (a starting point to measure, not a tuned value), no
  `reasoning_effort`, no JSON-schema `response_format`. Force any of them with `--workers`,
  `--reasoning-effort`, and `--response-schema`.
- `--extra-body '<json>'` merges a JSON object into every request's `extra_body`. Use it for
  server-specific switches, for example turning Qwen3 thinking off or guided decoding. The
  value is recorded in the summary as `config.extra_body`.
- Structured output: the system prompt asks for one JSON object and the Pydantic validation with
  feedback retries is the guarantee on every backend. Constrained decoding only reduces retries;
  a server that rejects the field fails every record (the breaker aborts after 25 failures).
- A refused or unreachable connection is fatal for local (not retried for minutes).
- Prompt longer than the model's context: that record fails as permanent and the run continues.
  Lower `--max-input-chars` (about 24000 for a 12k-token window).
- Provenance: each record stores `backend`. Mixing records from a different model, prompt
  version, schema version, or backend raises `ProvenanceMismatchError` unless you pass
  `--allow-mixed`. Records written before this feature count as `voyager`.
- Stopping: a stopped or crashed local server aborts the run with exit 3, keeps the finished
  records, and the same command resumes once the server is back.
- One GPU: the LLM server and the embedding model share one GPU (12 GB) and must not run at the
  same time. Stop the LLM server before any embedding command, and stop the embedding process
  before starting the server.

### L-1 Pick a server

vLLM gives the best throughput with several concurrent requests, Ollama is the easiest to
install, and the llama.cpp server is the lightest. Install it yourself, in WSL2 or Windows. The
GPU is an RTX 5070 (Blackwell): the CUDA build of the server must support compute capability
12.0 (inferred: that needs a recent release built for CUDA 12.8 or newer).

Pass: the server starts and sees the GPU (`nvidia-smi` lists it).

### L-2 Download the model

Recommended: Qwen3-8B, 4-bit. The weights are about 5 to 6 GB (inferred: 8.2 billion parameters
at roughly 4.5 bits). Qwen publishes `Qwen/Qwen3-8B-AWQ` and 4-bit GGUF files (inferred; verify
the exact repository names). On Ollama the tag `qwen3:8b` is a 4-bit build (inferred).
Full-precision Qwen3-8B is about 16 GB and will not fit.

Pass: the files are on disk and free VRAM before loading is near 12 GB.

### L-3 Start the server

Use a context limit of about 12,288 tokens and a GPU-memory share that leaves headroom. Inferred
vLLM example: `vllm serve Qwen/Qwen3-8B-AWQ --max-model-len 12288 --gpu-memory-utilization 0.90`
(check `vllm serve --help`). Inferred estimate: the KV cache costs about 0.14 MB per token in
16-bit, so two 8k-token requests need about 2.3 GB, which is why the local default is 2 workers.
If WSL2 and the server are on different sides (Windows and WSL), `localhost` may not reach it
(mirrored networking or a tunnel may be needed); the curl below checks this.

Pass: `curl http://localhost:8000/v1/models` lists the model id (adjust the port).

### L-4 Configure .env

Add `LLM_BACKEND=local`, `LOCAL_LLM_BASE_URL=http://localhost:8000/v1`, and
`LOCAL_LLM_MODEL=<the id from /v1/models>`. Leave `LOCAL_LLM_API_KEY` unset unless you started
the server with a key.

Pass: `git status` still does not list `.env`.

### L-5 Preflight

```
uv run structurer --source resumes --backend local --preflight
```

Pass: prints a JSON line with `"found": true` for the model. A refused connection exits 3 with a message naming the
URL (the server is not running or the URL is wrong).

### L-6 Five-record trial

Make sure Qwen3 thinking is off, then:

```
uv run structurer --source resumes --backend local --limit 5 --max-input-chars 24000 \
  --out-dir data/structured_trial_local/resumes
```

How to switch thinking off depends on the server (inferred: vLLM accepts
`--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'`; Ollama and llama.cpp use
their own switches, or append `/no_think` in a system-prompt copy). If thinking stays on, replies
start with a think block. The parser strips only a complete leading `<think>...</think>`; if
`max_tokens` runs out inside an unterminated think block, nothing is stripped. Truncated thinking
shows up as "empty reply" or "not valid JSON" failures ("empty reply" appears when the content
itself is empty, for example when the server puts reasoning in a separate field).

Pass: 5 of 5 succeed, no empty-reply or not-valid-JSON failures, and titles and skills match the clean text.

### L-7 Sample of 100

Use a trial directory that holds no Voyager records, for each source, timed with `time`:

```
time uv run structurer --source <source> --backend local --sample 100 --seed 0 \
  --max-input-chars 24000 --out-dir data/structured_trial_local/<source>
```

If the server rejects `response_format` you will see 400-style permanent failures: re-run
without `--response-schema` (the local default). Compare failure rates and a few records with the
Voyager trial.

Pass: failures under 2 percent and seconds per record for each source.

### L-8 Project the full run

Inferred order of magnitude: a record needs about 2.5k input tokens and 0.6k output tokens, so
37,265 records may take roughly 10 to 15 hours on this GPU; measure, do not trust. Decide one
model for the whole corpus: records from different models or backends are refused when mixed
(`ProvenanceMismatchError`) unless you pass `--allow-mixed`, and mixing a local 8B with a
Voyager 27B would make the dataset non-uniform. Record the server name, version, and
quantization in your project notes (records store only the model id and backend).

Pass: a written plan for which backend produces the final dataset.

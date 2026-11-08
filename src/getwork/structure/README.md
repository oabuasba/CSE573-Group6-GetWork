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
else means revise the prompt as a new version (do not edit v1). Time the run with the wall clock (for example `time uv run structurer ...`) and note
`rate_limited` to project a full run (37,265 calls, which can take a day or more). Estimate the
full output size (trial JSONL size x N/100); each committed file must stay under 100 MB.

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

# Web app runbook

FastAPI backend (`webapp serve`, `webapp check`) and a React UI (`src/web`). It matches resumes to postings with the
three systems (GetWork skill clusters, embeddings only, keywords only) and explains each match. Explanations
(`POST /api/explain`) are always rules-based and never call a language model. The chat agent
(`src/getwork/agent/README.md`, `POST /api/chat`, the "Chat about this suggestion" button) is the only LLM feature. The default is
`--explainer auto`: the chat agent is enabled when the LLM settings resolve (the same
`VOYAGER_API_KEY`, `VOYAGER_BASE_URL` and `VOYAGER_MODEL`, or `LLM_BACKEND=local` with `LOCAL_LLM_*`, that the structurer
uses; `--backend`, `--model` and `--env-file` apply) and `langgraph` is installed. Otherwise the server starts with chat off, prints one `chat agent disabled: ...` notice on stderr naming the missing variables (never values),
and shows the reason as `chat.reason` in `/api/health`. `--explainer agent` requires the agent (exit 2 if it cannot
start); `--explainer template` turns chat off (`chat.reason` says so). `/api/health` has `explainer` (always `template`) and
`chat` (`enabled`, `model`, `backend`, `prompt_version`, `reason`); the UI shows the chat button when `chat.enabled` is true.
`webapp check` prints the same health JSON (rules-based explanations plus the chat state) and calls no model.

Which artifacts are served is your choice, never fixed in code:

- `--artifacts-dir PATH` (both subcommands) wins over the environment variable `WEBAPP_ARTIFACTS_DIR`, which wins over
  the default `artifacts_trial`.
- PATH may be a run folder (it holds `alias/`, `documents/`, `taxonomy/`, plus `relabeled/` and `retrieval/` from the
  pipeline) such as `artifacts_trial/v5`, or a folder of versioned runs such as `artifacts_trial`. For the second
  form the highest complete `vN` subfolder is used (v10 beats v9; a vN missing a required folder is skipped and the
  skip is logged).
- The run actually loaded is shown in `webapp check` (`artifacts_dir`), in `/api/health` and in the startup line
  `Artifacts: ...`. The agent reads its skill vocabulary from the same resolved folder.

## WA-1 demo run

```
uv run webapp serve --demo        # http://127.0.0.1:8000, API docs at /docs
uv run webapp check --demo
```

Pass: `check` prints health JSON with `"mode": "demo"` and exits 0; the server prints its address.

## WA-2 trial artifacts check and serve

```
uv run webapp check --artifacts-dir artifacts_trial            # or artifacts_trial/v5, or set WEBAPP_ARTIFACTS_DIR
uv run webapp serve --artifacts-dir artifacts_trial
```

`check` loads everything and prints health without starting a server, calling the language model, or loading the
embedding model. Pass: exit 0 and health shows `"mode": "artifacts"`, the resolved `artifacts_dir`, counts, systems
and versions. Exit 2 means a missing, stale or mismatched artifact; the message names the fix (for example
`doc-embed build` or `doc-embed embed`). Without a key or GPU the server still starts: corpus search works, and the
form and text options show why they are unavailable (`capabilities` in health).

## WA-3 text query with your own keys

Text queries are structured by a language model and embedded locally. Set the variable names (never commit values)
`VOYAGER_API_KEY`, `VOYAGER_BASE_URL`, `VOYAGER_MODEL` (or `LLM_BACKEND` with the local settings) in `.env` or the
environment, and for embeddings `EMBED_BACKEND` and `EMBED_MODEL` (local needs `uv sync --extra embed`).

```
uv run webapp serve --artifacts-dir artifacts_trial --model MODEL --embed-backend local --embed-model MODEL
```

The embedding model loads on the first form or text query that needs it, never at startup, and queries are embedded
one at a time. If the GPU is shared with another job, start the server only when the memory is free, or use
`--no-embed` (corpus and keywords-only search still work). The embedding model must be the one the stored vectors
were made with, else the query fails with `embedding_mismatch`. Privacy: pasted text goes to the configured
language-model service for structuring; it is not stored or logged by the server.

## WA-4 full corpus

```
uv run webapp serve --artifacts-dir artifacts --systems Full,B1+
```

A vector set of 28,842 resumes at 2,560 dimensions is about 295 MB; `--systems` loads fewer sets (B1+ needs none).
Pass: health counts match the corpus size.

## WA-5 building the UI and single-process run

```
npm --prefix src/web ci
npm --prefix src/web run build
uv run webapp serve --demo
```

Pass: `http://127.0.0.1:8000/` shows the app and `/docs` shows the API. For frontend development run
`npm --prefix src/web run dev` (a mock API by default; set `GETWORK_REAL_API=1` to proxy to the real server, and
`GETWORK_API_HOST` / `GETWORK_API_PORT` to change its address, default 127.0.0.1 and 8000). Tests: `npm --prefix src/web test`.

"Chat about this suggestion" is a link in an expanded result card that opens a separate chat page (`#/chat/<id>`). The last
results and the conversations are kept only in the browser tab (`sessionStorage`) and are lost when the tab closes; a new search
clears the conversations and "Start over" clears one. "Back to results" restores the results with the same card open. To try the
chat page run `uv run webapp serve` with chat enabled; the `npm run dev` mock server has no `/api/chat` endpoint. The footer follows
the content (at the bottom of short pages, after it on long ones). Layout (sticky footer, one-row controls, the composer staying at
the bottom of a long conversation, Back and Forward) cannot be checked in jsdom and is a visual check you run yourself.

## WA-6 troubleshooting

- Frontend not built: `/` shows "Frontend not built"; run the build above.
- Embedding mismatch (409): the query embedder differs from the stored vectors; pass the same `--embed-backend` and
  `--embed-model` used for `doc-embed embed`.
- Capability unavailable (503): read `capabilities` in `webapp check`; the reason names the missing setting, such as
  `VOYAGER_API_KEY`, or the flag that disabled it (`--no-structure`, `--no-embed`).
- No "Chat about this suggestion" button: the button needs `chat.enabled` true. Run `uv run webapp check --demo`
  (or `--artifacts-dir ...`) and read `chat`: `enabled: false` with a `reason`, and the stderr notice, name the missing
  setting (for example `VOYAGER_API_KEY`, `VOYAGER_MODEL`) or `uv sync --extra agent`. It is also absent with
  `--explainer template`.
- Public host warning: `--host` other than 127.0.0.1 binds to a public address and there is no authentication and no
  CORS; keep the default unless you trust the network.
- Run selection: the resolver picks the highest `vN` that has `alias/`, `documents/` and `taxonomy/`; if that run lacks a
  later stage (for example vectors), loading fails with an error naming the fix instead of falling back to an older run.
- Artifacts not found: the error names the folder; pass the run folder itself (for example `artifacts_trial/v5`).

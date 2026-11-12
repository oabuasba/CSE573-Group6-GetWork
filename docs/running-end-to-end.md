# Running GetWork end to end

One runbook from a fresh clone to the web UI. Every command, flag, variable name and path here was checked against the code
(`--help` of each command, the settings modules, `scripts/run_pipeline.sh`). Per-stage detail lives in the stage READMEs, linked
in each section. No secret values or .env contents appear here (only variable names for secrets); never commit `.env`.

Legend: **[offline]** needs no key, network or GPU. **[Voyager]** makes real, billed API calls with your key: run it yourself. **[GPU]**
needs the local embedding model (`uv sync --extra embed`). Cost and run-time figures appear only where the stage READMEs give them.

## Contents

1. [Quick start](#1-quick-start-the-minimum-path-to-see-the-ui)
2. [Prerequisites and install](#2-prerequisites-and-install)
3. [Configuration (`.env`)](#3-configuration-env)
4. [Input data](#4-input-data)
5. [Overview diagram](#5-overview-diagram)
6. [The pipeline, stage by stage](#6-the-pipeline-stage-by-stage)
7. [Artifact folders and versions](#7-artifact-folders-and-versions)
8. [Running the web UI](#8-running-the-web-ui)
9. [The match agent](#9-the-match-agent)
10. [Verification and troubleshooting](#10-verification-and-troubleshooting)

## 1. Quick start: the minimum path to see the UI

Needs `uv` and Node >= 20.11. No key, no data, no GPU.

```
git clone https://github.com/oabuasba/CSE573-Group6-GetWork.git && cd CSE573-Group6-GetWork
uv sync
npm --prefix src/web ci
npm --prefix src/web run build          # writes src/web/dist
uv run webapp check --demo              # prints health JSON
uv run webapp serve --demo              # http://127.0.0.1:8000, API docs at /docs
```

Pass: `check` prints JSON with `"mode": "demo"` and exits 0; `serve` prints `GetWork is serving on http://127.0.0.1:8000` and
`Artifacts: demo data`, and `/` shows the app. Without `npm run build` the API still works and `/` shows "Frontend not built".

Then on real artifacts (you already have a `vN` run, for example in `artifacts_trial/`):

```
uv run webapp check --artifacts-dir artifacts_trial      # resolves the highest complete vN
uv run webapp serve --artifacts-dir artifacts_trial
```

Pass: health shows `"mode": "artifacts"` and the resolved `artifacts_dir` (for example `artifacts_trial/v5`). Searching from an
example document needs no key (section 8). If you have no artifacts yet, follow sections 4 to 7.

## 2. Prerequisites and install

| Need | Detail |
| --- | --- |
| Python >= 3.11 | `requires-python` in `pyproject.toml`; `uv` installs it for you if missing |
| [uv](https://docs.astral.sh/uv/) | runs every command as `uv run ...` |
| git, Linux or WSL | the structurer runbook states Linux or WSL |
| Node >= 20.11 and npm | `engines.node` in `src/web/package.json`; only for the React UI |
| ASU Voyager API key | only for [Voyager] steps (section 6). The portal needs an ASURITE login and the ASU network or VPN |

```
uv sync                      # core dependencies plus the default dev group (pytest, httpx, langgraph)
uv sync --extra embed        # + sentence-transformers and torch (several GB): local embeddings [GPU]
uv sync --extra agent        # + langgraph: the LLM match agent
```

Extras in `pyproject.toml`:

| Extra | Installs | Needed for |
| --- | --- | --- |
| (core) | fastapi, uvicorn, pandas, pyarrow, numpy, scikit-learn, scipy, umap-learn, plotly, matplotlib, openai, pydantic, python-dotenv, langdetect | everything except the two below |
| `embed` | sentence-transformers, torch | the `local` embedding backend (`EMBED_BACKEND=local` is the default). Not needed with `--embed-backend voyager` |
| `agent` | langgraph | `webapp serve` (the default `--explainer auto` needs it for chat). `langgraph` is also in the `dev` group, which a plain `uv sync` installs, so the extra matters for `--no-dev` installs |

Combine extras in one command if you want both (`uv sync --extra embed --extra agent`); each `uv sync` makes the environment match the
flags given, so repeat the extras every time.

GPU notes (from `src/getwork/embed/README.md`): the local model is `Qwen/Qwen3-Embedding-4B` (first use downloads it from Hugging Face,
about 8 GB, inferred in that README). The documented card is a 12 GB RTX 5070 (Blackwell, which that README infers needs a PyTorch
build for CUDA 12.8 or newer). Check with `uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`.
A local LLM server and the local embedding model do not fit a 12 GB card together: stop one before starting the other.

Console commands registered by `uv sync`: `structurer`, `alias-table`, `embed-preflight`, `webapp`, `taxonomy`, `relabel`, `doc-embed`,
`retrieve`.

## 3. Configuration (`.env`)

```
cp .env.example .env     # git ignores .env; fill it in yourself
```

Commands that build an LLM or embedding client read `.env` (override with `--env-file PATH`): `structurer`, `alias-table propose`, `taxonomy embed`/`name`, `relabel`, `doc-embed embed`, `embed-preflight`, `webapp`. `retrieve` and `taxonomy counts`/`grid`/`select`/`build`/`plot` do not. Real environment variables win over the file. Tests and all [offline]
steps need none of it. These are all the variable names the code reads (from `getwork/llm/settings.py`, `getwork/embed/settings.py`,
`getwork/webapp/cli.py`):

| Variable | Used for | Notes |
| --- | --- | --- |
| `VOYAGER_API_KEY` | Voyager LLM and Voyager embeddings | required when the backend is voyager |
| `VOYAGER_BASE_URL` | same | code enforces `https://` (or `http://` for localhost, 127.0.0.1, ::1); `.env.example` shows a value ending in `/v1`, which is the convention to follow |
| `VOYAGER_MODEL` | Voyager LLM model id | list with `uv run structurer --source resumes --preflight` |
| `LLM_BACKEND` | structuring backend: `voyager` (default) or `local` | `--backend` on the command line wins |
| `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL` | local OpenAI-compatible LLM server | required when the backend is local |
| `LOCAL_LLM_API_KEY` | local server key | optional, only if the server needs one |
| `EMBED_BACKEND` | embedding backend: `local` (default) or `voyager` | `--embed-backend` wins |
| `EMBED_MODEL` | local embedding model id | default `Qwen/Qwen3-Embedding-4B`; `--embed-model` wins |
| `VOYAGER_EMBED_MODEL` | Voyager embedding model id | no default; `--embed-model` wins |
| `WEBAPP_ARTIFACTS_DIR` | folder the web app serves | `--artifacts-dir` wins (section 8) |

Two choices to make once:

- **Structuring / LLM backend** (structurer, `taxonomy name`, `relabel --use-llm`, `alias-table propose`, text queries in the web app,
  the agent): `voyager` needs the three `VOYAGER_*` variables; `local` needs `LOCAL_LLM_BASE_URL` and `LOCAL_LLM_MODEL` and any
  OpenAI-compatible server (vLLM, Ollama, llama.cpp). Set it with `LLM_BACKEND` or `--backend`. Details and a server checklist:
  [`src/getwork/structure/README.md`](../src/getwork/structure/README.md). Do not mix models or backends in one dataset (the
  structurer refuses with `ProvenanceMismatchError` unless `--allow-mixed`).
- **Embedding backend**: `local` (default; Qwen3-Embedding-4B, needs `--extra embed`, no key) or `voyager` (needs `VOYAGER_API_KEY`,
  `VOYAGER_BASE_URL` and a model id from `VOYAGER_EMBED_MODEL` or `--embed-model`). Options: `--embed-backend`, `--embed-model`,
  `--device` (auto, cpu, cuda, cuda:N), `--dtype`, `--embed-batch-size`, `--max-seq-length`.
  Check one with `uv run embed-preflight [--embed-backend voyager --embed-model ID]`.
- **The embedding model that built the vectors must also embed typed queries.** Backend, model and dimension are stored in each
  vector meta file and compared; the same model id served by local and Voyager is still refused. `scripts/run_pipeline.sh` embeds
  with `--embed-backend voyager --embed-model qwen3-embedding-8b`, so a server over its artifacts needs those same two flags
  (and the Voyager key and URL) for form and text queries. Otherwise the API returns `embedding_mismatch`.

## 4. Input data

Only the raw inputs are git-ignored; you place them yourself. The outputs of preprocessing (`data/processed/{resumes,postings}/`) and of the structurer (`data/structured/{resumes,postings}/`) are tracked in git, so a fresh clone can skip sections 6.1 and 6.2 and go straight to `scripts/run_pipeline.sh` (only the embed steps and `taxonomy name` call a model). Paths below are the defaults of the preprocessing commands.

| Source | Place at | Format the code expects |
| --- | --- | --- |
| Resumes (Jiechieu and Tsopze corpus) | `data/resume_corpus/resumes_corpus.zip` | a zip of `.txt` resumes, each with a same-named `.lab` label file; labels must be among the ten known resume classes or preprocessing raises `ValueError` |
| Postings (LinkedIn Job Postings, Kaggle) | `data/job_postings/` | `companies/postings.csv`, `jobs/job_industries.csv`, `mappings/industries.csv` under that folder |

## 5. Overview diagram

```
 data/resume_corpus/resumes_corpus.zip        data/job_postings/{companies,jobs,mappings}/*.csv
            |                                              |
            v                                              v
   run_resumes   [offline]                       run_postings   [offline]
            +--------------------+-------------------------+
                                 v
                  data/processed/{resumes,postings}/      *_clean.parquet
                                 |
                                 v
             structurer   [Voyager or local LLM]           (long run; resumable)
                                 |
                                 v
                  data/structured/{resumes,postings}/     *_structured.jsonl
                                 |
 ---------- scripts/run_pipeline.sh (ARTIFACTS_DIR, e.g. artifacts_trial/v6) ----------
                                 v
   alias-table build, apply  [offline]    -> alias/, canonical/
   taxonomy counts           [offline]    -> skills/skill_counts.parquet
   taxonomy embed            [Voyager|GPU]-> skills/skill_vectors.npy
   taxonomy grid, select     [offline]    -> taxonomy/grid_results.csv, selection.json
   taxonomy name             [LLM]        -> taxonomy/cluster_names.csv  (+ fit and overlap reviews)
   taxonomy build, plot      [offline]    -> taxonomy/taxonomy.json, plots/
   relabel                   [offline]    -> relabeled/<source>/
   doc-embed build           [offline]    -> documents/<source>/*_documents.*
   doc-embed embed           [Voyager|GPU]-> documents/<source>/*_full.npy, *_b2.npy
   retrieve run              [offline]    -> retrieval/topk_*.parquet
 ------------------------------------------------------------------------------------
                                 |
                                 v
   uv run webapp serve --artifacts-dir ...   (needs alias/, documents/, taxonomy/)
                                 |
                                 v
                 FastAPI backend + React UI (src/web/dist) at http://127.0.0.1:8000
                 default --explainer auto (LLM chat when LLM settings are set; explanations stay rules-based)
```

## 6. The pipeline, stage by stage

Run from the repo root. `structurer` reads `data/processed/<source>` and writes `data/structured/<source>`; `relabel` and `doc-embed build` default `--processed-dir` to `data/processed`. Only `alias-table` defaults to the trial folder `data/structured_trial`, and every artifact stage defaults to `artifacts_trial`; the script and your real runs should pass paths explicitly. Each stage README has options, exit codes and checklists.

### 6.1 Preprocessing [offline]

```
uv run python -m getwork.preprocess.run_resumes     # --zip PATH --out DIR   (defaults: the zip above -> data/processed/resumes)
uv run python -m getwork.preprocess.run_postings    # --data DIR --out DIR   (defaults: data/job_postings -> data/processed/postings)
```

Outputs per source in `data/processed/<source>/`: `<source>_clean.parquet`, `<source>_raw.parquet`, `<source>_drop_log.csv`,
`<source>_summary.json` (postings also `postings_domain_review.csv`). Pass: exit 0 and a
summary JSON printed.

### 6.2 Structurer [Voyager or local LLM]

```
uv run structurer --source resumes --preflight                    # contacts the LLM service; prints model info, exit 3 if not found
uv run structurer --source resumes --dry-run                      # no calls
uv run structurer --source resumes --limit 5 --out-dir data/structured_trial/resumes      # trial
uv run structurer --source resumes                                # full run -> data/structured/resumes (resumable: rerun the same command)
uv run structurer --source postings
uv run structurer --source resumes --backend local --preflight    # local server instead of Voyager
```

Reads `data/processed/<source>/`. Outputs in `data/structured/<source>/`: `<source>_structured.jsonl` (checkpoint),
`.parquet`, `_structured_failures.csv`, `_structured_summary.json`. Exit codes: 0 complete, 1 some records failed (retried on rerun),
2 usage/config, 3 aborted or preflight failed, 130 interrupted. README note: a full run is 37,265 calls and may take a day or more.
Full options and the 8-step user checklist: [`src/getwork/structure/README.md`](../src/getwork/structure/README.md).

### 6.3 Later stages: use the script or run them one by one

After structuring, everything through retrieval is one command (see 6.14):

```
ARTIFACTS_DIR=artifacts_trial/v6 STRUCTURED_DIR=data/structured bash scripts/run_pipeline.sh
```

Or run the stages yourself with the commands below (always pass the same `--artifacts-dir`; the stage defaults are `artifacts_trial` and,
for `alias-table`, `--structured-dir data/structured_trial`).

### 6.4 Alias table [offline]

```
uv run alias-table build --structured-dir data/structured --artifacts-dir ROOT
uv run alias-table apply --structured-dir data/structured --artifacts-dir ROOT
```

Outputs: `ROOT/alias/{alias_table.json, alias_build_report.json, alias_version_report.json}` and
`ROOT/canonical/<source>/<source>_canonical.jsonl`, `.parquet`, `_canonical_summary.json`. The table carries a content version
(`sha256:<12 hex>`); every later stage refuses to mix versions. Optional `alias-table propose --model ID` drafts alias pairs with an
LLM [Voyager or local] and writes a draft CSV only. [`src/getwork/alias/README.md`](../src/getwork/alias/README.md).

### 6.5 Embedding client and preflight [offline for fake/tests; GPU or Voyager for real]

`getwork.embed` is the shared client behind `taxonomy embed` and `doc-embed embed` (section 3). Check your setup before a long embed:

```
uv run embed-preflight                                            # local backend
uv run embed-preflight --embed-backend voyager --embed-model ID   # Voyager
```

Pass: a JSON report with `norms_ok` true and `related_cosine` greater than `unrelated_cosine`.
[`src/getwork/embed/README.md`](../src/getwork/embed/README.md).

### 6.6 Skill taxonomy: counts, embed, grid, select [counts/grid/select offline]

```
uv run taxonomy counts --artifacts-dir ROOT                       # --top-n 1000 --min-df 2 by default
uv run taxonomy embed  --artifacts-dir ROOT --embed-backend voyager --embed-model qwen3-embedding-8b   # [Voyager] (or local [GPU])
uv run taxonomy grid   --artifacts-dir ROOT                       # UMAP + DBSCAN/HDBSCAN/k-means grid; exit 2 if none eligible
uv run taxonomy select --artifacts-dir ROOT --config-id kmeans-01 # optional; choose another id from grid_results.csv
```

Outputs: `ROOT/skills/{skill_counts.parquet, counts_summary.json, skill_vectors.npy, skill_vectors_ids.json, skill_vectors_meta.json,
_embed_cache/}`, `ROOT/taxonomy/{grid_context.json, grid_results.csv, grid_labels.npy, selection.json, review_<config_id>.csv}`.
Embedding is chunked (256) and cached, so an interrupted run resumes on rerun. The first UMAP call compiles with numba and can take
about a minute.

### 6.7 Cluster naming [`taxonomy name` itself defaults to `--method heuristic`; `run_pipeline.sh` defaults to `llm`]

```
uv run taxonomy name --artifacts-dir ROOT --method llm --workers 4 --fit-rounds 1 --overlap-rounds 2 --force   # [Voyager or local LLM]
uv run taxonomy name --artifacts-dir ROOT --method heuristic --force                             # [offline] member-list names
uv run taxonomy build --artifacts-dir ROOT                        # add --allow-member-names after heuristic naming
uv run taxonomy plot  --artifacts-dir ROOT                        # [offline] ROOT/taxonomy/plots/index.html
```

`name` writes `taxonomy/cluster_names.csv` and refuses to overwrite an existing file without `--force`, so hand edits are safe until you
force it. Edit names by hand in that file, then run `build`, which writes `taxonomy/taxonomy.json`. Options that exist (see
[`src/getwork/taxonomy/README.md`](../src/getwork/taxonomy/README.md)):

- **Parallel naming**: `--workers N` (default 1; the script uses 4). Results are the same for any N >= 2 but can differ from 1.
- **Fit review**: `--fit-rounds N` (default 1, `0` turns it off) runs first. One LLM judge call lists the model-made names that do not
  describe their members; each is asked once for a better name with the judge's reason (prompt `refit_v1`). Renamed rows have
  `prompt_version` `name_v2+refit_v1`; a cluster that cannot be fixed keeps its name with status `weak_fit` (not fatal). The summary
  adds `n_fit_renamed` and `n_weak_fit` after `n_overlap`. `taxonomy build` only warns for each `weak_fit` row (and prints `n_weak_fit`);
  set the status to `ok` after fixing a name. Renaming cannot fix a cluster with mixed members.
- **Overlap review**: `--overlap-rounds N` (default 2, `0` turns it off) re-asks clusters whose names are too close, by word rules plus
  one LLM judge call; `--no-overlap-judge` uses the rules only. Clusters that stay overlapping get status `overlap` (not fatal).
- Naming sends `max_tokens` 1024 (`--max-tokens`) with reasoning effort `none` for Voyager (`--reasoning-effort`); a reasoning model
  with a small budget can return empty replies. Exit 1 means some clusters fell back to heuristic names; fix them by hand before `build`.

### 6.8 Relabeling [offline by default]

```
uv run relabel --artifacts-dir ROOT --processed-dir data/processed             # lookup-only: free, no calls
uv run relabel --artifacts-dir ROOT --use-llm --model ID --source resumes      # optional [LLM]; then --source postings, same model
```

Reads alias table, taxonomy and canonical records. Outputs `ROOT/relabeled/<source>/<source>_relabeled.{jsonl,parquet}`,
`_relabeled_summary.json`, `<source>_relabel_failures.csv`. Use the same mode for both sources.
[`src/getwork/relabel/README.md`](../src/getwork/relabel/README.md).

### 6.9 Document embedding

```
uv run doc-embed build --artifacts-dir ROOT --processed-dir data/processed     # [offline] texts, levels (bands 2,5)
uv run doc-embed embed --artifacts-dir ROOT --embed-backend voyager --embed-model qwen3-embedding-8b   # [Voyager] or local [GPU]
```

`build` outputs `ROOT/documents/<source>/<source>_documents.{jsonl,parquet}` and `_documents_summary.json`. `embed` adds
`<source>_full.npy`, `_full_ids.json`, `_full_meta.json` and the same three for `b2`; rerun the same command after an interruption (chunks
are cached; `--no-cache` skips the cache). Exit 3 means an embedding error with cached chunks kept.
[`src/getwork/docembed/README.md`](../src/getwork/docembed/README.md).

### 6.10 Retrieval (two-way) [offline]

```
uv run retrieve run  --artifacts-dir ROOT                         # --systems all|B1+,B2,Full  --k 10
uv run retrieve show --artifacts-dir ROOT --direction resume_to_posting --id ID --system Full --k 5
```

Outputs `ROOT/retrieval/topk_{b1plus,b2,full}_{resume_to_posting,posting_to_resume}.parquet` and `retrieval_summary.json`.
`--systems B1+` needs only the documents tables. `show` computes one query from stored vectors (no API calls).
[`src/getwork/retrieval/README.md`](../src/getwork/retrieval/README.md). Also `scripts/match_random.sh` (picks a random resume and shows
B1+, B2 and Full, no API calls): `[ARTIFACTS_DIR=...] [K=5] bash scripts/match_random.sh [RESUME_ID]`.

### 6.11 `scripts/run_pipeline.sh` (everything after structuring)

```
[STRUCTURED_DIR=...] [ARTIFACTS_DIR=...] [PROCESSED_DIR=...] [CLUSTER_CONFIG_ID=...] \
[NAME_METHOD=llm|heuristic] [NAME_WORKERS=4] [NAME_OVERLAP_ROUNDS=2] [NAME_FIT_ROUNDS=1] bash scripts/run_pipeline.sh [CONFIG_ID]
```

It runs 13 commands in order and stops at the first failure (`set -euo pipefail`); each is echoed as `==> <command>`. It does not run
preprocessing, the structurer, or the web app. It changes to the repo root, so use absolute paths for the variables to avoid surprises.

| Setting | Default | Meaning |
| --- | --- | --- |
| `STRUCTURED_DIR` | `data/structured` | read by `alias-table build` and `apply` |
| `ARTIFACTS_DIR` | `artifacts` | output tree for every step. **Set it** (for example `artifacts_trial/v6`) to land where the web app's default looks |
| `PROCESSED_DIR` | `data/processed` | read by `relabel` and `doc-embed build` |
| `CONFIG_ID` argument or `CLUSTER_CONFIG_ID` | `kmeans-01` (k=10, exactly 10 clusters, no noise) | what `taxonomy select --config-id` picks right after `grid`; the argument wins; grid's own pick is always overridden |
| `NAME_METHOD` | `llm` | `heuristic` is for offline runs and adds `--allow-member-names` to `taxonomy build` |
| `NAME_WORKERS` | `4` | passed as `taxonomy name --workers` |
| `NAME_OVERLAP_ROUNDS` | `2` | passed as `--overlap-rounds` (`0` disables) |
| `NAME_FIT_ROUNDS` | `1` | passed as `--fit-rounds` (`0` disables) |

Steps: `alias-table build`, `apply`, `taxonomy counts`, `embed`, `grid`, `select`, `name ... --force`, `build`, `plot`, `relabel`,
`doc-embed build`, `doc-embed embed`, `retrieve run`. Embedding steps always pass `--embed-backend voyager --embed-model
qwen3-embedding-8b`, so the embed steps need `VOYAGER_API_KEY` and `VOYAGER_BASE_URL` in `.env` (`VOYAGER_EMBED_MODEL` is not needed). With the default
`NAME_METHOD=llm`, step 7 (`taxonomy name`) also needs an LLM configuration, because the script cannot pass `--model`: either `VOYAGER_MODEL`, or
`LLM_BACKEND=local` with `LOCAL_LLM_BASE_URL` and `LOCAL_LLM_MODEL`. Without one, the run fails at step 7 with `missing required setting(s)`
*after* the billed `taxonomy embed` already ran. `NAME_METHOD=heuristic` needs no LLM variables. **Billed Voyager calls happen at
`taxonomy embed`, `doc-embed embed` and, with `NAME_METHOD=llm` on the voyager backend, `taxonomy name` including the fit and overlap reviews (the fit review adds one judge call per round plus one to three per weak cluster after naming; with
`LLM_BACKEND=local` the naming calls go to your local server). Run the script yourself.** Before overwriting, an existing `selection.json` and `cluster_names.csv` are copied to
`$ARTIFACTS_DIR/taxonomy/archive/<name>_<UTC timestamp>.<ext>`.

Examples:

```
bash scripts/run_pipeline.sh                                    # artifacts/, kmeans-01
bash scripts/run_pipeline.sh kmeans-04                          # another configuration
ARTIFACTS_DIR=artifacts_trial/v6 bash scripts/run_pipeline.sh 2>&1 | tee artifacts_trial/v6/pipeline_run.txt
```

Log convention (`pipeline_run.txt`): the script itself writes no log. This repo's agent rules (`.claude/agents/orchestrator.md`) save the
full console output of a run to `artifacts_trial/vN/pipeline_run.txt` (next unused `vN`, never overwriting an earlier log). Do the same
with `tee` as above (create the folder first: `mkdir -p artifacts_trial/v6`).

Embed caching: each chunk's key covers its index, texts and backend/model; an identical rerun re-embeds nothing, a different model or a
deleted `_embed_cache/` re-embeds everything. Each Voyager embed step still sends one probe request at start-up. `taxonomy grid` exiting 2
(no eligible configuration) stops the script. Because `name --force` overwrites hand edits, run the stages individually if you edit names.

## 7. Artifact folders and versions

A run folder (`ROOT`, for example `artifacts_trial/v5`):

```
ROOT/
  alias/        alias_table.json + reports                      required by the web app
  canonical/    <source>/ alias-applied records
  skills/       skill counts, vectors, _embed_cache
  taxonomy/     grid files, selection.json, cluster_names.csv, taxonomy.json, plots/   required by the web app
  relabeled/    <source>/ records with a cluster per skill
  documents/    <source>/ texts and the Full/B2 vector files                            required by the web app
  retrieval/    topk_*.parquet, retrieval_summary.json
  pipeline_run.txt   (only if you saved the log)
```

- Required to start the web app: `alias/`, `documents/`, `taxonomy/`. If the documents lack vectors, loading fails with an error naming the
  fix (`doc-embed embed`) instead of falling back to an older run. `retrieval/` is not needed to serve (the app ranks on demand).
- The repo keeps experiment runs side by side as `artifacts_trial/v1` ... `v5` (git-ignored; `artifacts/` is too). A run is just a
  folder: pick the next unused `vN` as `ARTIFACTS_DIR` so earlier runs stay comparable.
- Resolution (web app): if the path you give holds all three required folders, it is used as is. Otherwise the highest `vN` subfolder
  that has all three is used (v10 beats v9; incomplete ones are skipped and logged). A path with neither is an error naming the folder.
- Changing the alias table or taxonomy invalidates everything after it (versions are stamped on every artifact); rerun the later stages
  into a fresh `vN` rather than mixing.

## 8. Running the web UI

### 8.1 Build the frontend, or use the dev server

```
npm --prefix src/web ci
npm --prefix src/web run build      # src/web/dist; `webapp serve` serves it at /
npm --prefix src/web test           # frontend tests (vitest)
```

Dev server (hot reload) in a second terminal:

```
npm --prefix src/web run dev                       # mock API by default
GETWORK_REAL_API=1 npm --prefix src/web run dev    # proxy /api to a running `webapp serve`
# GETWORK_API_HOST (default 127.0.0.1) and GETWORK_API_PORT (default 8000) change the proxy target
```

### 8.2 Demo mode (no key, no data)

```
uv run webapp check --demo
uv run webapp serve --demo
```

Pass: health has `"mode": "demo"`; demo mode has no language model (`structure` unavailable), and its `embed` capability is demo scoring.

### 8.3 Real artifacts

```
uv run webapp check --artifacts-dir artifacts_trial      # or a run: artifacts_trial/v5, or set WEBAPP_ARTIFACTS_DIR
uv run webapp serve --artifacts-dir artifacts_trial      # --host (default 127.0.0.1) --port (default 8000) --static-dir
```

Precedence: `--artifacts-dir` beats `WEBAPP_ARTIFACTS_DIR`, which beats the default `artifacts_trial`. `check` loads everything and prints
health JSON without a server, any model call, or loading the embedding model. Pass: exit 0, `"mode": "artifacts"`, the resolved
`artifacts_dir`, counts, systems and versions; `serve` also prints `Artifacts: <resolved folder>`. Exit 2 means a missing, stale or
mismatched artifact; the message names the fix. Useful flags: `--systems B1+,B2,Full` (load fewer vector sets; B1+ needs none),
`--b1plus-weights FILE`, `--max-query-chars` (default 20000), `--no-structure`, `--no-embed`, `--host` other than loopback prints a
warning (no authentication, no CORS).

### 8.4 What works with and without a key

`capabilities` in the `check` / `/api/health` output has two entries, `structure` (text to skills) and `embed` (query vector), each with
`available`, plus a `reason` when unavailable (and a `detail` when available). For `structure` the reason names the missing variables. For `embed`, `check` reports `available: true` unless
`--no-embed` is given: it does not validate the embedding configuration, so a missing key or missing `embed` extra only shows up as a 503
(`capability_unavailable`) on the first query that needs the embedder.

| Query | Needs | Without a key |
| --- | --- | --- |
| Example / corpus document (the UI's examples; `kind: corpus`) | the stored vectors only | works, all three systems |
| Form (title, skills, years, level) with the keywords-only system (B1+) | nothing | works |
| Form with Full or B2 | `embed`: the same embedding model as the stored vectors | `capability_unavailable` unless embedding is configured; with Voyager needs the key |
| Pasted text (`kind: text`) | `structure` (LLM key or local server) always; `embed` only for Full or B2 (not B1+) | `capability_unavailable` |

To enable text queries: set the LLM variables (section 3) and start with the embedding model that matches the stored vectors, for example
for `scripts/run_pipeline.sh` artifacts:

```
uv run webapp serve --artifacts-dir artifacts_trial --model MODEL --embed-backend voyager --embed-model qwen3-embedding-8b
```

For artifacts built with the local model use `--embed-backend local --embed-model Qwen/Qwen3-Embedding-4B` (needs `--extra embed`). The
embedding model loads on the first query that needs it, not at startup, and queries are embedded one at a time. If the GPU is shared, use
`--no-embed` (corpus and B1+ search still work). Pasted text goes to the configured LLM service; the server does not store or log it.
The full runbook is [`src/getwork/webapp/README.md`](../src/getwork/webapp/README.md).

## 9. The match agent

Match explanations (`POST /api/explain`, "Why this match?") are always rules-based and never call a model. The LangGraph agent is the
per-match chat only (`--explainer auto` by default): it answers follow-up questions with claims verified against the two
documents (the "Chat about this suggestion" button). It makes real model calls with your key,
using the model in `VOYAGER_MODEL`, the same as the structurer (so `--model` is not needed). If `VOYAGER_API_KEY`, `VOYAGER_BASE_URL`
or `VOYAGER_MODEL` (or `LOCAL_LLM_*` with `LLM_BACKEND=local`) is missing, or `langgraph` is not installed, the server prints one
`chat agent disabled: ...` notice naming the variables and starts with chat off; `/api/health` then shows `chat.enabled: false` and `chat.reason`.

```
uv sync --extra agent
uv run webapp check --demo                                          # no model call; health shows explainer "template" plus chat (enabled with the model, or disabled with a reason)
uv run webapp serve --demo
uv run webapp serve --artifacts-dir artifacts_trial
uv run webapp serve --artifacts-dir artifacts_trial --explainer agent   # strict: exit 2 if the agent cannot start
uv run webapp serve --artifacts-dir artifacts_trial --explainer template   # chat off
```

Flags: `--explainer {auto,agent,template}` (default `auto`), `--agent-backend {voyager,local}`, `--agent-model ID`,
`--agent-timeout SECONDS` (default 60). By default the agent uses the structuring backend and model (`--backend`, `--model`); the
`--agent-*` flags override them. `--agent-backend local` reads `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL`, `LOCAL_LLM_API_KEY`. The agent
reads its skill vocabulary from the resolved run folder.

"Chat about this suggestion" is a link in an expanded result card that opens a separate chat page (`#/chat/<id>`). The last results and
the conversations are kept only in the browser tab (`sessionStorage`) and are lost when the tab closes; a new search clears the
conversations, and "Start over" clears one. "Back to results" restores the results with the same card open. To try the chat page, run
`uv run webapp serve` with chat enabled (the variables above); the `npm run dev` mock server has no `/api/chat` endpoint, so the page
shows its unavailable state there. The page layout (footer at the bottom of short pages, one-row controls, the composer staying at the
bottom of a long conversation, Back and Forward) cannot be checked in jsdom and needs a visual check by you in a real browser. The footer follows the content.

Failure behaviour: if the model is unreachable, rejects the request, times out, or its answer cannot be verified, the chat shows a short
plain-language message; the match explanation is separate and always rules-based. Restart with `--explainer template` to
remove the chat. Only structured fields (titles, levels, skills, clusters, ranks, scores, the conversation) go to the model,
never raw resume or posting text. Full runbook (MA-1 to MA-6): [`src/getwork/agent/README.md`](../src/getwork/agent/README.md).

## 10. Verification and troubleshooting

```
uv run pytest -q                     # Python tests (src/tests), all offline with fake clients
npm --prefix src/web test            # frontend tests
git status --porcelain data/ artifacts_trial/ src/web/dist     # tests should create nothing here
```

Pass: both suites exit 0 with no key, model or network.

| Symptom or message | Cause and fix |
| --- | --- |
| `uv` reports a stale lock, or tests fail on a new dependency | `uv.lock` is out of date after a `pyproject.toml` change: run `uv lock` online, then `uv sync`. Offline, run `uv run --offline --frozen pytest -q` (documented in the specs; the lock is currently consistent: `uv lock --check --offline` passes) |
| `langgraph is not installed; run: uv sync --extra agent` | install the agent extra (also installed by plain `uv sync` through the dev group) |
| `missing required setting(s): VOYAGER_API_KEY, ...` | `.env` lacks those names (the message lists names, never values); check `--env-file` and `--backend` |
| No "Chat about this suggestion" button | the server started with chat off: read the `chat agent disabled: ...` stderr notice, or `chat.reason` in `uv run webapp check`; set the named variables (`VOYAGER_API_KEY`, `VOYAGER_BASE_URL`, `VOYAGER_MODEL`) or install langgraph. Not shown either with `--explainer template` |
| `POST /api/chat` returns 502 | read the one `agent failed: reason=<r> detail=<...>` line on the server console (the 502 JSON also has `error.reason`): `model_rejected` means the service refused the request (see the status and message in `detail`: 401/403/404 key, URL or model; 400/422 an unsupported option such as `--extra-body`), `llm_unavailable` means it could not be reached, `invalid_reply` means the model's answer was unusable, and `unverified` means no claim could be checked against the two documents (answers where only some claims fail are shown without them, with a note). Keys, bearer tokens and URLs are redacted. See `src/getwork/agent/README.md` |
| `/` shows "Frontend not built" | run `npm --prefix src/web ci` then `npm --prefix src/web run build` |
| `artifacts directory not found` / `is not a complete artifacts run: missing folders ...` | pass the run folder itself (for example `artifacts_trial/v5`) or build the missing stages (`alias-table build`, `taxonomy build`, `doc-embed build`); the web app needs `alias/`, `documents/`, `taxonomy/` |
| `webapp check` exits 2 naming `doc-embed build` or `doc-embed embed` | the chosen (highest complete) `vN` lacks that stage; run it, or pass an older complete run explicitly |
| `embedding_mismatch` (HTTP 409) | the query embedder differs from the stored vectors; pass the `--embed-backend` and `--embed-model` used for `doc-embed embed` |
| `capability_unavailable` (HTTP 503) | read `capabilities` in `webapp check`; for `structure` the reason names the missing setting (such as `VOYAGER_API_KEY`) or the flag (`--no-structure`). `check` does not validate the embedding setup, so an embed problem (missing key, missing `embed` extra) appears only as a 503 on the first Full/B2 query |
| Embedding model out of memory, or "reducing batch size" | stop any local LLM server; lower `--embed-batch-size` (for example 4); or use `--embed-backend voyager` |
| `taxonomy grid` exits 2 (no eligible configuration) | raise `--max-noise` or pass `--grid-file`; the script stops here |
| `taxonomy name` exit 1, or `taxonomy build` refuses member-list names | some clusters fell back to heuristic names: edit `cluster_names.csv` by hand, or `build --allow-member-names` for offline runs |
| `ProvenanceMismatchError`, `RelabelProvenanceError` or "alias table version" errors | records from different models, backends or table versions were mixed; rerun the stage into a fresh `vN`, or `--allow-mixed` (not for reported data) |
| Structurer exit 3 / preflight not found | wrong `VOYAGER_MODEL` or base URL (must end in `/v1`); a 404 `{"detail":"Not Found"}` means a legacy key to rotate |
| Public host warning | `--host` other than 127.0.0.1 has no authentication; keep the default |

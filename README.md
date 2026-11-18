# GetWork

GetWork (CSE 573, Group 6) is a two-way resume and job matcher for software engineering and adjacent fields. Given a resume it returns the top-k job postings; given a posting it returns the top-k candidates. The full design builds a skill taxonomy from the corpus so that related skills (React and Vue) earn partial credit, then has an agent explain each match. See `docs/proposal.md` and `specs/architecture.html`.

New here? [`docs/running-end-to-end.md`](docs/running-end-to-end.md) is the single runbook from a fresh clone to the web UI.

## Status

| Stage | State |
| --- | --- |
| Resume and posting preprocessing (`run_resumes`, `run_postings`) | Built |
| Structurer (LLM extraction via ASU Voyager) | Built |
| Alias table, skill taxonomy, relabeling | Built |
| Document embeddings (`doc-embed`) | Built |
| Retrieval (B1+, B2, Full) | Built |
| Web app (FastAPI backend, React UI, deterministic explainer; try `webapp serve --demo`) | Built |
| LLM explanation agent and chat (default in `webapp serve` when the LLM settings are present; `src/getwork/agent`) | Built |
| Evaluation harness (LLM judges, metrics, report; `src/getwork/evaluation`), preliminary run done (`eval/preliminary_results`); full run and human evaluation pending | Built |

## Repository structure

```
data/raw/job_postings/      LinkedIn Job Postings CSVs (postings.csv is git-ignored, see data/raw/README.md)
data/raw/resume_corpus/     IT resume corpus (zips and resume_samples.txt are git-ignored)
data/processed/             cleaned resumes and postings (preprocess output)
data/structured/            LLM-structured records (structurer output)
data/pipeline_run_v7/       the v7 pipeline run: alias, canonical, skills, taxonomy, relabeled, documents, retrieval
eval/preliminary_results/   the evaluation run reported in the slides
eval/other/                 earlier and side evaluation runs (v6, v6.2, v7 heldout and smoke)
scripts/bash/               run_pipeline.sh, run_evaluation.sh, run_evaluation_preliminary.sh, match_random.sh
scripts/python/slides/      deck generator and its layout and contrast checks
scripts/python/other/       peek_dev_metrics.py
src/getwork/                the Python packages; src/web is the React UI; src/tests the tests
docs/  specs/               proposal, runbook, deviations, specs
artifacts/, artifacts_trial/  new and older trial pipeline runs (git-ignored)
```

## Quick start

```
uv sync
uv run pytest                                                   # offline tests
uv run webapp serve --demo                                      # demo data, no keys needed
uv run webapp serve --artifacts-dir data/pipeline_run_v7 --embed-backend voyager --embed-model qwen3-embedding-8b   # the committed v7 run
```

The v7 run needs its four document vector files (`documents/*/*.npy`, about 620 MB), which are git-ignored. Get them from a teammate or rebuild them once with a Voyager key (about 45 minutes; see [`data/pipeline_run_v7/README.md`](data/pipeline_run_v7/README.md)):

```
uv run doc-embed embed --artifacts-dir data/pipeline_run_v7 --embed-backend voyager --embed-model qwen3-embedding-8b
```

Data flows left to right: `data/raw` -> `data/processed` (`run_resumes`, `run_postings`) -> `data/structured` (`structure`, Voyager LLM) -> `data/pipeline_run_v7` (`bash scripts/bash/run_pipeline.sh`) -> `eval/` (`bash scripts/bash/run_evaluation_preliminary.sh`, see [`eval/README.md`](eval/README.md)).

## Architecture

```
 data/raw/resume_corpus/resumes_corpus.zip      data/raw/job_postings/*.csv
              |                                       |
              v                                       v
   [BUILT] run_resumes                     [BUILT] run_postings
        (filter, dedup)                  (domain filter, filter, dedup)
              |                                       |
              +------------------+--------------------+
                                 v
                     data/processed/{resumes,postings}/
                                   |
                                   v
                   [BUILT] structurer (ASU Voyager LLM)
                                   |
                                   v
                     data/structured/{resumes,postings}/
                                   |
                                   v
   [BUILT] alias table, skill taxonomy, relabel
                                   |
                                   v
   [BUILT] document embeddings -> two-way retrieval
                                   |
                                   v
   [BUILT] evaluation (preliminary run done)
                                   |
                                   v
   [BUILT] FastAPI backend + React UI + template explainer (real artifacts or --demo)
   [BUILT, optional extra] LLM explanation agent + chat (default when LLM settings are set)
```

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)
- git
- An ASU Voyager API key, only for real structurer runs, the Voyager embed steps (`taxonomy embed`, `doc-embed embed`), the evaluation judges and the web app chat. The Voyager portal, where you create the key, needs an ASURITE login and the ASU network or VPN. Tests and preprocessing do not need it.
- Raw data: small files are committed, the large ones are git-ignored and you place them yourself (see [`data/raw/README.md`](data/raw/README.md)):
  - Resumes: `data/raw/resume_corpus/resumes_corpus.zip` (the Jiechieu and Tsopze corpus).
  - Postings (LinkedIn Job Postings, Kaggle): under `data/raw/job_postings/`, the preprocessor reads `companies/postings.csv`, `jobs/job_industries.csv` and `mappings/industries.csv`.

## Install

```
git clone https://github.com/oabuasba/CSE573-Group6-GetWork.git && cd CSE573-Group6-GetWork
uv sync                      # core dependencies plus the dev group (pytest, httpx)
uv run pytest                # tests live in src/tests
```

`uv sync` needs Python >= 3.11 and installs the dependencies from `pyproject.toml` (fastapi, uvicorn, pandas, pyarrow, numpy, scikit-learn, scipy, umap-learn, plotly, matplotlib, openai, pydantic, python-dotenv, langdetect) and registers the commands `structurer`, `alias-table`, `taxonomy`, `relabel`, `doc-embed`, `retrieve`, `evaluate`, `embed-preflight` and `webapp`.

Optional extra `embed` (local embeddings; installs torch and sentence-transformers, several GB):

```
uv sync --extra embed
```

It is needed only for the default local embedding backend. With `--embed-backend voyager` it is not needed. See [`src/getwork/embed/README.md`](src/getwork/embed/README.md).

Secrets: copy `.env.example` to `.env` and fill in `VOYAGER_API_KEY`, `VOYAGER_BASE_URL` and `VOYAGER_MODEL` (and `VOYAGER_EMBED_MODEL` if you use Voyager embeddings). Tests and offline steps do not need it.

## Run the web app

Web frontend (optional; Node >= 20.11):

```
npm --prefix src/web install
npm --prefix src/web run build      # writes src/web/dist, which `webapp serve` picks up
uv run webapp serve --demo          # http://127.0.0.1:8000, API docs at /docs
```

`webapp serve` also takes `--host`, `--port` and `--static-dir`; `webapp check --demo` prints health. Frontend tests: `npm --prefix src/web test`.

Run it on your artifacts (the folder is your choice: `--artifacts-dir`, else the environment variable `WEBAPP_ARTIFACTS_DIR`, else `artifacts_trial`; a folder of `vN` runs resolves to its highest complete run, or pass a run such as `data/pipeline_run_v7`):

```
uv run webapp check --artifacts-dir data/pipeline_run_v7 --embed-backend voyager --embed-model qwen3-embedding-8b
uv run webapp serve --artifacts-dir data/pipeline_run_v7 --embed-backend voyager --embed-model qwen3-embedding-8b
```

Pass the same embedding backend and model that built the stored vectors (v7: Voyager `qwen3-embedding-8b`). The default is the local model, and with a mismatch typed resumes and postings fail with an embedding-mismatch error (409).

See the runbook [`src/getwork/webapp/README.md`](src/getwork/webapp/README.md).

Match explanations are always rules-based (no model call). The per-match chat agent ("Chat about this suggestion") is the default (`--explainer auto`). They use the same LLM settings and model as the structurer (`VOYAGER_API_KEY`, `VOYAGER_BASE_URL`, `VOYAGER_MODEL` in `.env`, or `LLM_BACKEND=local` with `LOCAL_LLM_*`) and need `langgraph` (`uv sync --extra agent`, or a plain `uv sync`). The model calls are made by you, with your own key; see `src/getwork/agent/README.md`. If a setting is missing the server prints a `chat agent disabled: ...` notice and runs with chat off (no chat button):

```
uv run webapp serve --demo
# optional: --explainer template (chat off) | agent (fail if it cannot start), --model MODEL (default VOYAGER_MODEL),
#           --agent-backend voyager|local, --agent-model MODEL, --agent-timeout SECONDS (default 60)
```

## Preprocess and structure

Each preprocessing step writes to `data/processed/<source>/` by default:

```
uv run python -m getwork.preprocess.run_resumes     # --zip PATH --out DIR
uv run python -m getwork.preprocess.run_postings    # --data DIR --out DIR
```

Structurer setup:

```
uv run structurer --source resumes --preflight    # contacts Voyager (needs key, network and VOYAGER_MODEL); prints info for that model, exits 3 if not found
```

The full command list, exit codes and output files are in [`src/getwork/structure/README.md`](src/getwork/structure/README.md).

## Pipeline script (structured records to retrieval)

`scripts/bash/run_pipeline.sh` runs every step after structuring, in order, stopping at the first failure (`set -euo pipefail`). It does not run preprocessing, the structurer, or the web app. Each step is echoed as `==> <command>`.

```
[STRUCTURED_DIR=...] [ARTIFACTS_DIR=...] [PROCESSED_DIR=...] [CLUSTER_CONFIG_ID=...] bash scripts/bash/run_pipeline.sh [CONFIG_ID]
```

Before the first run on the full data (10,567 resumes, 8,416 postings in v7), `preprocess` and `structure` must already have produced `STRUCTURED_DIR` for both sources. The structurer writes to `data/structured/<source>` by default.

Environment variables (all optional):

| Variable | Default | Used by |
| --- | --- | --- |
| `STRUCTURED_DIR` | `data/structured` | `alias-table build` and `apply` only (`--structured-dir`) |
| `ARTIFACTS_DIR` | `artifacts` (a fresh tree, separate from `artifacts_trial/`) | every step (`--artifacts-dir`) |
| `PROCESSED_DIR` | `data/processed` | `relabel` and `doc-embed build` (`--processed-dir`) |
| `TOP_N` | unset (command default, 1000) | `taxonomy counts --top-n` (use a fresh `ARTIFACTS_DIR` when you set it) |
| `NAME_METHOD` | `llm` | `taxonomy name --method` (`heuristic` is offline-only and adds `--allow-member-names` to `taxonomy build`) |
| `NAME_WORKERS` | `4` | `taxonomy name --workers` |
| `NAME_OVERLAP_ROUNDS` | `2` | `taxonomy name --overlap-rounds` (`0` disables) |
| `NAME_FIT_ROUNDS` | `1` | `taxonomy name --fit-rounds` (`0` disables) |

- `CONFIG_ID` (optional first argument, for example `kmeans-04`) or the environment variable `CLUSTER_CONFIG_ID`: the configuration that `taxonomy select --config-id` picks right after `taxonomy grid`. The default is `kmeans-01` (k=10, so exactly 10 clusters and no noise); the argument wins over the variable. `grid`'s own pick in `selection.json` is always overridden (the previous file is archived, see below). Pick other ids from `$ARTIFACTS_DIR/taxonomy/grid_results.csv`.
- The script changes into the repo root itself, so it can be run from anywhere, but relative `STRUCTURED_DIR`, `ARTIFACTS_DIR` and `PROCESSED_DIR` resolve against the repo root, not your current directory (use absolute paths to avoid surprises). The embed steps need `.env` (you provide it; not in git) with `VOYAGER_API_KEY` and `VOYAGER_BASE_URL`. With the default `NAME_METHOD=llm`, step 7 also needs `VOYAGER_MODEL` (or `LLM_BACKEND=local` with `LOCAL_LLM_BASE_URL` and `LOCAL_LLM_MODEL`), because the script cannot pass `--model`; without it the run fails at step 7 after the billed `taxonomy embed`. `NAME_METHOD=heuristic` needs no LLM variables. `VOYAGER_EMBED_MODEL` is commented out in `.env.example` and is not needed here, because the script passes `--embed-backend voyager --embed-model qwen3-embedding-8b`.
- Inputs per step: steps 1-2 read `STRUCTURED_DIR`; steps 3-9 and 13 read only `ARTIFACTS_DIR`; step 10 (`relabel`) also reads `PROCESSED_DIR` (the clean parquet per source); step 11 (`doc-embed build`) also reads `PROCESSED_DIR/postings/postings_clean.parquet` for listed experience levels (if it is missing, posting levels are derived from years of experience and a note is printed); step 12 reads `ARTIFACTS_DIR` only.
- Archive behavior: `taxonomy name --force` overwrites `cluster_names.csv`, and `taxonomy grid` and `select` overwrite `selection.json`. Before those steps the script copies an existing file to `$ARTIFACTS_DIR/taxonomy/archive/<name>_<UTC timestamp>.<ext>` (for example `cluster_names_20261109T060353Z.csv`), and skips quietly if the file does not exist.

| Step | Command | Main output under `$ARTIFACTS_DIR/` |
| --- | --- | --- |
| 1 | `alias-table build` | `alias/alias_table.json` |
| 2 | `alias-table apply` | `canonical/<source>/` |
| 3 | `taxonomy counts` | `skills/skill_counts.parquet` |
| 4 | `taxonomy embed` (Voyager call) | `skills/skill_vectors.npy` |
| 5 | `taxonomy grid` | `taxonomy/grid_results.csv`, `taxonomy/selection.json` |
| 6 | `taxonomy select` (always; default `kmeans-01`) | `taxonomy/selection.json` |
| 7 | `taxonomy name --method $NAME_METHOD --workers $NAME_WORKERS --overlap-rounds $NAME_OVERLAP_ROUNDS --fit-rounds $NAME_FIT_ROUNDS --force` (default `llm`, real LLM calls) | `taxonomy/cluster_names.csv` |
| 8 | `taxonomy build` | `taxonomy/taxonomy.json` |
| 9 | `taxonomy plot` (offline) | `taxonomy/plots/` (`index.html`, `cluster_map.png`, `cluster_sizes.png`, ...) |
| 10 | `relabel` (lookup-only mode) | `relabeled/<source>/` |
| 11 | `doc-embed build` | `documents/<source>/<source>_documents.{jsonl,parquet}`, `<source>_documents_summary.json` |
| 12 | `doc-embed embed` (Voyager call) | `documents/<source>/<source>_full.npy`, `<source>_full_ids.json`, `<source>_full_meta.json`, and the same three for `b2` |
| 13 | `retrieve run` | `retrieval/topk_*.parquet`, `retrieval_summary.json` |

Warnings:

- Steps 4, 7 (with the default `NAME_METHOD=llm`) and 12 make real Voyager calls. Step 7 uses `--force`, so it overwrites `cluster_names.csv`, including hand edits; the previous copy is archived as described above. If you edit names, restore from the archive or run steps individually instead (see [`src/getwork/taxonomy/README.md`](src/getwork/taxonomy/README.md)).
- `taxonomy grid` exits 2 when no configuration is eligible, which stops the script.
- `embed-preflight` and `structurer --preflight` exit 3 when the model is not found or the preflight request fails.

Embed caching (steps 4 and 12):

- Texts are embedded in chunks of 256 (`--chunk-size`), and each chunk is saved under an `_embed_cache` folder: `$ARTIFACTS_DIR/skills/_embed_cache` for `taxonomy embed`, `$ARTIFACTS_DIR/documents/<source>/_embed_cache` for `doc-embed embed`. `doc-embed embed --no-cache` turns it off (the script does not pass it).
- A chunk's key is its index, its texts, and the backend/model info. An identical rerun, or a rerun after an interrupted run, embeds no cached chunk. Each Voyager embed step still sends one probe request at start-up, when the client is built.
- Inserting or removing a text shifts every later chunk, so those chunks are embedded again. Changing the model or backend, or deleting the cache folder, re-embeds everything.

Examples:

```
bash scripts/bash/run_pipeline.sh                       # full data, 10 clusters (kmeans-01)
bash scripts/bash/run_pipeline.sh kmeans-04             # force a specific configuration
ARTIFACTS_DIR=artifacts_trial/v8 bash scripts/bash/run_pipeline.sh   # write a separate run
```

After it finishes, inspect results with `uv run retrieve show --direction resume_to_posting --id ID --artifacts-dir "$ARTIFACTS_DIR"` (without the flag it reads `artifacts_trial/`). Per-stage options are in the READMEs: [`alias`](src/getwork/alias/README.md), [`taxonomy`](src/getwork/taxonomy/README.md), [`relabel`](src/getwork/relabel/README.md), [`docembed`](src/getwork/docembed/README.md), [`retrieval`](src/getwork/retrieval/README.md).

### Try a random resume

`scripts/bash/match_random.sh` picks a random resume id from `$ARTIFACTS_DIR/documents/resumes/resumes_documents.parquet` and runs `retrieve show` (direction `resume_to_posting`) for B1+, B2 and Full. It uses stored vectors, so it makes no API calls.

```bash
[ARTIFACTS_DIR=...] [K=5] bash scripts/bash/match_random.sh [RESUME_ID]
```

## Evaluation

The `evaluate` command samples queries, pools the top 10 from B1+, B2 and Full, has two LLM judges on Voyager (Gemma 4 31B and Llama 4 Maverick 17B by default) label every pair blind, and computes P@5, P@10, MRR, nDCG@10, judge agreement and ARI. Only the judge stages call Voyager, and they refuse to start unless `USER_RUN_JUDGE=1` is set. Every other stage is offline.

```
# preliminary run (no B1+ tuning, no human check); use a fresh EVAL_DIR each time
USER_RUN_JUDGE=1 ARTIFACTS_DIR=data/pipeline_run_v7 EVAL_DIR=eval/run \
  bash scripts/bash/run_evaluation_preliminary.sh

# full run (adds B1+ weight tuning on the dev queries)
USER_RUN_JUDGE=1 ARTIFACTS_DIR=data/pipeline_run_v7 EVAL_DIR=eval/run bash scripts/bash/run_evaluation.sh
```

Results land in `EVAL_DIR`: `report.md`, `metrics.json`, `agreement.json`, `ari.json`. The run in the slides is `eval/preliminary_results/` (30 + 30 test queries).

Human check (pending): the two authors label `pairs_blind_human.jsonl` independently, 0/1/2, saved as `human/<author>.jsonl` in the run folder. Then:

```
uv run evaluate agreement --scope human --artifacts-dir data/pipeline_run_v7 --out-dir eval/preliminary_results
uv run evaluate metrics --labels <author> --queries human --artifacts-dir data/pipeline_run_v7 --out-dir eval/preliminary_results
```

Stages, the rubric and the label rules are in [`eval/README.md`](eval/README.md) and [`src/getwork/evaluation/README.md`](src/getwork/evaluation/README.md).

## Development workflow (multi-agent)

Features are built by a plan, build, review pipeline of Claude Code agents defined in `.claude/agents/`. You talk only to the orchestrator.

### Using it

```
claude --agent orchestrator
```

Ask for a feature in plain language (for example, "add the skill selection stage"). The orchestrator reads `specs/status.json` at session start and summarizes where things stand, so you can also ask "what's running?". It answers from that file, not from memory. To shape `specs/architecture.html` yourself before any build work, you can start the planner directly with `claude --agent planner`.

| State | Where |
| --- | --- |
| Task and feature progress (phase, deps, fix rounds, worktrees) | `specs/status.json` |
| One build brief per feature (goal, ACs, tasks) | `specs/features/<feature>.html` |
| Shared contracts, decision log (ADRs), feature index | `specs/architecture.html` |

### Agents

| Agent | Model | Job | Writes |
| --- | --- | --- | --- |
| orchestrator | Sonnet 5.5 | Routes requests, relays questions, manages worktrees and branches, tracks state. Never writes code or specs. | `specs/status.json`, status column of the feature index, git worktrees and branches |
| planner | Sonnet 5.5 | Turns a request into a spec with testable ACs and a task breakdown; pushes back on weak designs. | `specs/` only |
| developer | Sonnet 5.5 | Implements one task (`T-n`) and its tests in its own worktree. | The task's files, uncommitted |
| reviewer | Sonnet 5.5 | Independently runs the tests and checks the task against the spec's ACs and code style. Read-only. | Nothing (returns a verdict) |
| researcher | Sonnet 5.5 | Looks up external facts (such as API docs) with cited sources. Read-only. | Nothing (returns answers) |

### Adding a feature

```
 request
    |
    v
 planner writes spec <--------+
    |                         |
    +-- needs-answers/pushback --> you answer --+
    | ready
    v
 [you approve]  (always stops here)
    |
    v
 build: tasks in dependency order, up to 3 developers in parallel
    |   (one worktree + branch per task)
    v
 reviewer (fresh, independent)
    |-- fail --> same developer fixes (max 2 rounds, then escalates to you)
    |-- pass
    v
 integrate: commit, merge into feature/<feature>
    |
    v
 all tasks merged -> feature done -> you review and merge
```

1. **Request.** The orchestrator passes your request to the planner.
2. **Spec.** The planner returns `ready`, `needs-answers` or `pushback` (concern, why, alternative). The orchestrator relays the latter two to you and sends your reply to the same planner, until it is `ready`.
3. **Approval.** You get the spec path, a summary, the task list and assumptions. Nothing is built until you approve.
4. **Build.** Each task whose dependencies are merged gets a worktree off `feature/<feature>` and a developer. Tasks the planner flagged `large` are put to you first.
5. **Review.** A fresh reviewer checks each finished task, seeing only the developer's deviations, not its report.
6. **Fix loop.** On `fail`, the findings go back to the same developer, then a new review. After 2 failed rounds the orchestrator brings it to you. Spec problems go back to the planner.
7. **Integrate.** The orchestrator asks before committing and merging a passed task (unless you tell it to auto-merge for the session). When all tasks are merged the feature is `done`.

By default the orchestrator uses a worktree and branch per task. You can instead tell it to work in place on the current branch (e.g. `dev`) and treat a reviewer pass as the merge, which is how this repo's structurer was built.

Ground rules:

- The orchestrator never writes code or specs, and never pushes. It does not commit or merge to `main` without your say-so. You commit and push yourself, and you merge the finished feature branch.
- Real ASU Voyager calls and the `.env` file are user-only. No agent may read `.env` or any secrets file, and no agent may make real Voyager/LLM API calls (inference, or any key-authenticated request). The researcher's read-only doc lookups (WebSearch/WebFetch) are the exception. Tests use the fake client and offline fixtures. Run `structurer` against Voyager yourself.

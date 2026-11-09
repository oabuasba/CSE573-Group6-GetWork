# GetWork

GetWork (CSE 573, Group 6) is a two-way resume and job matcher for software engineering and adjacent fields. Given a resume it returns the top-k job postings; given a posting it returns the top-k candidates. The full design builds a skill taxonomy from the corpus so that related skills (React and Vue) earn partial credit, then has an agent explain each match. See `docs/proposal.md` and `specs/architecture.html`.

## Status

| Stage | State |
| --- | --- |
| Resume and posting preprocessing (`run_resumes`, `run_postings`) | Built |
| Structurer (LLM extraction via ASU Voyager) | Built |
| Alias table, skill taxonomy, relabeling | Built |
| Document embeddings (`doc-embed`) | Built |
| Retrieval (B1+, B2, Full) | Built |
| FastAPI backend and React UI (`webapp`, `src/web`) | Demo mode built; artifact mode and final checks pending |
| LLM explanation agent (only the template explainer exists) | Planned |
| Evaluation | Planned |

## Architecture

```
 data/resume_corpus/resumes_corpus.zip      data/job_postings/*.csv
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
   [PLANNED] evaluation
                                   |
                                   v
   [BUILT, demo data] FastAPI backend + React UI + template explainer
   [PLANNED] LLM explanation agent
```

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)
- git
- An ASU Voyager API key, only for real structurer runs and the Voyager embed steps (`taxonomy embed`, `doc-embed embed`) (Linux or WSL). The Voyager portal, where you create the key, needs an ASURITE login and the ASU network or VPN. Tests and preprocessing do not need it.
- Raw data, which git ignores and you place yourself:
  - Resumes: `data/resume_corpus/resumes_corpus.zip` (the Jiechieu and Tsopze corpus).
  - Postings (LinkedIn Job Postings, Kaggle): under `data/job_postings/`, the preprocessor reads `companies/postings.csv`, `jobs/job_industries.csv` and `mappings/industries.csv`.

## Install

```
git clone https://github.com/oabuasba/CSE573-Group6-GetWork.git && cd CSE573-Group6-GetWork
uv sync                      # core dependencies plus the dev group (pytest, httpx)
uv run pytest                # tests live in src/tests
```

`uv sync` needs Python >= 3.11 and installs the dependencies from `pyproject.toml` (fastapi, uvicorn, pandas, pyarrow, numpy, scikit-learn, scipy, umap-learn, plotly, matplotlib, openai, pydantic, python-dotenv, langdetect) and registers the commands `structurer`, `alias-table`, `taxonomy`, `relabel`, `doc-embed`, `retrieve`, `embed-preflight` and `webapp`.

Optional extra `embed` (local embeddings; installs torch and sentence-transformers, several GB):

```
uv sync --extra embed
```

It is needed only for the default local embedding backend. With `--embed-backend voyager` it is not needed. See [`src/getwork/embed/README.md`](src/getwork/embed/README.md).

Secrets: copy `.env.example` to `.env` and fill in `VOYAGER_API_KEY`, `VOYAGER_BASE_URL` and `VOYAGER_MODEL` (and `VOYAGER_EMBED_MODEL` if you use Voyager embeddings). Tests and offline steps do not need it.

Web frontend (optional; Node >= 20.11):

```
npm --prefix src/web install
npm --prefix src/web run build      # writes src/web/dist, which `webapp serve` picks up
uv run webapp serve --demo          # http://127.0.0.1:8000, API docs at /docs
```

`webapp serve` also takes `--host`, `--port` and `--static-dir`; `webapp check --demo` prints health. Only `--demo` works for now. Frontend tests: `npm --prefix src/web test`.

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

`scripts/run_pipeline.sh` runs every step after structuring, in order, stopping at the first failure (`set -euo pipefail`). It does not run preprocessing, the structurer, or the web app. Each step is echoed as `==> <command>`.

```
[STRUCTURED_DIR=...] [ARTIFACTS_DIR=...] [PROCESSED_DIR=...] bash scripts/run_pipeline.sh [CONFIG_ID]
```

Before the first run on the full data (4k resumes, 1k postings), `preprocess` and `structure` must already have produced `STRUCTURED_DIR` for both sources. The structurer writes to `data/structured/<source>` by default. If your structured output is still in the trial folder, move it yourself (the script never moves data):

```
git mv data/structured_trial data/structured     # or: mv data/structured_trial data/structured
```

Environment variables (all optional):

| Variable | Default | Used by |
| --- | --- | --- |
| `STRUCTURED_DIR` | `data/structured` | `alias-table build` and `apply` only (`--structured-dir`) |
| `ARTIFACTS_DIR` | `artifacts` (a fresh tree, separate from `artifacts_trial/`) | every step (`--artifacts-dir`) |
| `PROCESSED_DIR` | `data/processed` | `relabel` and `doc-embed build` (`--processed-dir`) |

- `CONFIG_ID` (optional, for example `kmeans-04`): if given, `taxonomy select --config-id CONFIG_ID` runs right after `taxonomy grid`. Without it, `grid` overwrites `selection.json` with its own pick (the previous file is archived, see below) and that pick is used. Pick the id from `$ARTIFACTS_DIR/taxonomy/grid_results.csv`.
- The script changes into the repo root itself, so it can be run from anywhere, but relative `STRUCTURED_DIR`, `ARTIFACTS_DIR` and `PROCESSED_DIR` resolve against the repo root, not your current directory (use absolute paths to avoid surprises). The embed steps need `.env` (you provide it; not in git) with `VOYAGER_API_KEY` and `VOYAGER_BASE_URL`. `VOYAGER_EMBED_MODEL` is commented out in `.env.example` and is not needed here, because the script passes `--embed-backend voyager --embed-model qwen3-embedding-8b`.
- Inputs per step: steps 1-2 read `STRUCTURED_DIR`; steps 3-8 and 12 read only `ARTIFACTS_DIR`; step 9 (`relabel`) also reads `PROCESSED_DIR` (the clean parquet per source); step 10 (`doc-embed build`) also reads `PROCESSED_DIR/postings/postings_clean.parquet` for listed experience levels (if it is missing, posting levels are derived from years of experience and a note is printed); step 11 reads `ARTIFACTS_DIR` only.
- Archive behavior: `taxonomy name --force` overwrites `cluster_names.csv`, and `taxonomy grid` and `select` overwrite `selection.json`. Before those steps the script copies an existing file to `$ARTIFACTS_DIR/taxonomy/archive/<name>_<UTC timestamp>.<ext>` (for example `cluster_names_20261109T060353Z.csv`), and skips quietly if the file does not exist.

| Step | Command | Main output under `$ARTIFACTS_DIR/` |
| --- | --- | --- |
| 1 | `alias-table build` | `alias/alias_table.json` |
| 2 | `alias-table apply` | `canonical/<source>/` |
| 3 | `taxonomy counts` | `skills/skill_counts.parquet` |
| 4 | `taxonomy embed` (Voyager call) | `skills/skill_vectors.npy` |
| 5 | `taxonomy grid` | `taxonomy/grid_results.csv`, `taxonomy/selection.json` |
| 6 | `taxonomy select` (only with `CONFIG_ID`) | `taxonomy/selection.json` |
| 7 | `taxonomy name --method heuristic --force` | `taxonomy/cluster_names.csv` |
| 8 | `taxonomy build` | `taxonomy/taxonomy.json` |
| 9 | `relabel` (lookup-only mode) | `relabeled/<source>/` |
| 10 | `doc-embed build` | `documents/<source>/<source>_documents.{jsonl,parquet}`, `<source>_documents_summary.json` |
| 11 | `doc-embed embed` (Voyager call) | `documents/<source>/<source>_full.npy`, `<source>_full_ids.json`, `<source>_full_meta.json`, and the same three for `b2` |
| 12 | `retrieve run` | `retrieval/topk_*.parquet`, `retrieval_summary.json` |

Warnings:

- Steps 4 and 11 make real Voyager calls.
- Step 7 uses `--force`, so it overwrites `cluster_names.csv`, including hand edits; the previous copy is archived as described above. If you edit names, restore from the archive or run steps individually instead (see [`src/getwork/taxonomy/README.md`](src/getwork/taxonomy/README.md)).
- `taxonomy grid` exits 2 when no configuration is eligible, which stops the script.
- `embed-preflight` and `structurer --preflight` exit 3 when the model is not found or the preflight request fails.

Embed caching (steps 4 and 11):

- Texts are embedded in chunks of 256 (`--chunk-size`), and each chunk is saved under an `_embed_cache` folder: `$ARTIFACTS_DIR/skills/_embed_cache` for `taxonomy embed`, `$ARTIFACTS_DIR/documents/<source>/_embed_cache` for `doc-embed embed`. `doc-embed embed --no-cache` turns it off (the script does not pass it).
- A chunk's key is its index, its texts, and the backend/model info. An identical rerun, or a rerun after an interrupted run, embeds no cached chunk. Each Voyager embed step still sends one probe request at start-up, when the client is built.
- Inserting or removing a text shifts every later chunk, so those chunks are embedded again. Changing the model or backend, or deleting the cache folder, re-embeds everything.

Examples:

```
bash scripts/run_pipeline.sh                       # full data, use the configuration grid picked
bash scripts/run_pipeline.sh kmeans-04             # force a specific configuration
STRUCTURED_DIR=data/structured_trial ARTIFACTS_DIR=artifacts_trial bash scripts/run_pipeline.sh   # trial tree
```

After it finishes, inspect results with `uv run retrieve show --direction resume_to_posting --id ID --artifacts-dir "$ARTIFACTS_DIR"` (without the flag it reads `artifacts_trial/`). Per-stage options are in the READMEs: [`alias`](src/getwork/alias/README.md), [`taxonomy`](src/getwork/taxonomy/README.md), [`relabel`](src/getwork/relabel/README.md), [`docembed`](src/getwork/docembed/README.md), [`retrieval`](src/getwork/retrieval/README.md).

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

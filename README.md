# GetWork

GetWork (CSE 573, Group 6) is a two-way resume and job matcher for software engineering and adjacent fields. Given a resume it returns the top-k job postings; given a posting it returns the top-k candidates. It builds a skill taxonomy from the corpus so that related skills (React and Vue) earn partial credit, then explains each match with rules and an optional LLM chat agent. Design: [`docs/proposal.md`](docs/proposal.md) and [`specs/architecture.html`](specs/architecture.html).

## Status

| Stage | State |
| --- | --- |
| Resume and posting preprocessing (`run_resumes`, `run_postings`) | Built |
| Structurer (LLM extraction via ASU Voyager) | Built |
| Alias table, skill taxonomy, relabeling | Built |
| Document embeddings (`doc-embed`) | Built |
| Retrieval (B1+, B2, Full) | Built |
| Web app (FastAPI backend, React UI, deterministic explainer; try `webapp serve --demo`) | Built |
| LLM explanation agent and chat (needs your own LLM settings; `src/getwork/agent`) | Built |
| Evaluation harness, preliminary run done (`eval/preliminary_results`); full run and human evaluation pending | Built |

## Requirements and install

- Python >= 3.11 and [uv](https://docs.astral.sh/uv/)
- Node >= 20.11 and npm, only for the web UI
- An ASU Voyager API key, only for real structurer runs, the Voyager embed steps (`taxonomy embed`, `doc-embed embed`), the evaluation judges and the web app chat. You create your own key and put it in your own environment (below). Tests, the demo and preprocessing do not need it.

```
git clone https://github.com/oabuasba/CSE573-Group6-GetWork.git && cd CSE573-Group6-GetWork
uv sync
```

`uv sync` installs the dependencies from `pyproject.toml` (including the chat agent's `langgraph`) and the commands `structurer`, `alias-table`, `taxonomy`, `relabel`, `doc-embed`, `retrieve`, `evaluate`, `embed-preflight` and `webapp`. The optional extra `uv sync --extra embed` (torch, several GB) is only for local embedding models; see [`src/getwork/embed/README.md`](src/getwork/embed/README.md).

Secrets: copy [`.env.example`](.env.example) to your own `.env` (or `export` the same variables in your shell) and fill in `VOYAGER_API_KEY`, `VOYAGER_BASE_URL` and `VOYAGER_MODEL`. Create your own and never commit it.

## Quick start (fresh clone, no keys)

```
uv run pytest                        # offline tests, about a minute
npm --prefix src/web ci              # the UI is built from source, so this step is needed once
npm --prefix src/web run build
uv run webapp serve --demo           # open http://127.0.0.1:8000 (API docs at /docs)
```

The demo uses six built-in resumes and postings. Without the two npm steps the API still runs (`/docs`) but there is no UI page. Without LLM settings the server prints `chat agent disabled: ...` and runs with chat off; match explanations stay rules-based. `uv run webapp check --demo` prints a health report. More options (`--host`, `--port`, `--explainer template|agent`, `--model`, `--agent-model`) are in [`src/getwork/webapp/README.md`](src/getwork/webapp/README.md); the chat agent is described in [`src/getwork/agent/README.md`](src/getwork/agent/README.md). Frontend tests: `npm --prefix src/web test`.

## Working with the committed data

| Folder | What it holds |
| --- | --- |
| `data/processed/` | cleaned resumes and postings (output of `run_resumes`, `run_postings`) |
| `data/structured/` | LLM-structured records (output of the structurer) |
| `data/pipeline_run_v7/` | the v7 pipeline run (alias, canonical, skills, taxonomy, relabeled, documents, retrieval) without the large embedding `.npy` files |
| `eval/preliminary_results/` | the preliminary evaluation run (judge labels, metrics, report) |

The big raw files and the document embeddings are not in git. [`data/raw/README.md`](data/raw/README.md) says where to download the raw datasets and where to place them; [`data/pipeline_run_v7/README.md`](data/pipeline_run_v7/README.md) lists the missing files and how to rebuild them (a Voyager key is needed).

Optional, after rebuilding the embeddings per that README (until then it stops with `file not found ... (run doc-embed embed)`):

```
uv run webapp check --artifacts-dir data/pipeline_run_v7 --embed-backend voyager --embed-model qwen3-embedding-8b
uv run webapp serve --artifacts-dir data/pipeline_run_v7 --embed-backend voyager --embed-model qwen3-embedding-8b
```

Pass the same embedding backend and model that built the stored vectors (v7: Voyager `qwen3-embedding-8b`). The default is the local model, and with a mismatch typed resumes and postings fail with an embedding-mismatch error (409).

Always pass `--artifacts-dir` (or set `WEBAPP_ARTIFACTS_DIR`) for real data; a pipeline script run writes its own folder, see below.

## Repository structure

```
data/raw/                   raw dataset folders (README says what to download)
data/processed/             cleaned resumes and postings
data/structured/            LLM-structured records
data/pipeline_run_v7/       v7 pipeline artifacts
eval/preliminary_results/   evaluation run reported in the slides
eval/other/                 earlier and side evaluation runs
scripts/bash/               pipeline and evaluation drivers
scripts/python/             slide generator and helper scripts
src/getwork/                Python packages (each has its own README)
src/web/                    React UI; src/tests the tests
docs/  specs/               proposal, runbook, deviations, specs
.claude/                    development agents and skills
```

Data flows left to right: `data/raw` -> `data/processed` (`run_resumes`, `run_postings`) -> `data/structured` (`structurer`) -> pipeline run (`scripts/bash/run_pipeline.sh`) -> `eval/` (`scripts/bash/run_evaluation_preliminary.sh`).

```
 resumes_corpus.zip + job_postings CSVs  -->  run_resumes / run_postings (filter, dedup)
   -->  structurer (ASU Voyager LLM)  -->  alias table, skill taxonomy, relabel
   -->  document embeddings  -->  two-way retrieval (B1+, B2, Full)
   -->  evaluation  /  FastAPI backend + React UI + template explainer (real artifacts or --demo)
        + optional LLM explanation agent and chat
```

More: [`scripts/README.md`](scripts/README.md), [`eval/README.md`](eval/README.md), [`docs/running-end-to-end.md`](docs/running-end-to-end.md) (runbook), and the stage READMEs: [`structure`](src/getwork/structure/README.md), [`alias`](src/getwork/alias/README.md), [`taxonomy`](src/getwork/taxonomy/README.md), [`relabel`](src/getwork/relabel/README.md), [`docembed`](src/getwork/docembed/README.md), [`retrieval`](src/getwork/retrieval/README.md), [`evaluation`](src/getwork/evaluation/README.md), [`webapp`](src/getwork/webapp/README.md), [`agent`](src/getwork/agent/README.md).

## Rerunning the pipeline (needs raw data and a Voyager key)

```
uv run python -m getwork.preprocess.run_resumes     # --zip PATH --out DIR
uv run python -m getwork.preprocess.run_postings    # --data DIR --out DIR
uv run structurer --source resumes --preflight      # contacts Voyager; see the structure README for the full run
ARTIFACTS_DIR=<new folder> bash scripts/bash/run_pipeline.sh [CONFIG_ID]
```

`run_pipeline.sh` runs every step after structuring (alias table, taxonomy, relabel, document embeddings, retrieval) and stops at the first failure. It reads `data/structured` (10,567 resumes and 8,416 postings in v7) and `data/processed`, writes only to `ARTIFACTS_DIR`, and defaults `CONFIG_ID` to `kmeans-01` (k=10). Pick a folder that is not an existing run. The embed steps and, by default, cluster naming make real Voyager calls. Optional variables: `STRUCTURED_DIR`, `PROCESSED_DIR`, `TOP_N`, `NAME_METHOD` (`heuristic` runs naming offline); all are documented at the top of the script. After it finishes, inspect results with `uv run retrieve show --direction resume_to_posting --id ID --artifacts-dir <that folder>`.

## Evaluation

`evaluate` samples queries, pools the top 10 from B1+, B2 and Full, has two Voyager LLM judges label every pair blind, and computes P@5, P@10, MRR, nDCG@10, judge agreement and ARI. The judge stages refuse to start unless you set `USER_RUN_JUDGE=1`; every other stage is offline. The preliminary run (no B1+ tuning, no human check) is `USER_RUN_JUDGE=1 ARTIFACTS_DIR=data/pipeline_run_v7 EVAL_DIR=<new folder> bash scripts/bash/run_evaluation_preliminary.sh` (use a fresh `EVAL_DIR` each time); the run in the slides is `eval/preliminary_results/`. Stages and label rules: [`eval/README.md`](eval/README.md) and [`src/getwork/evaluation/README.md`](src/getwork/evaluation/README.md).

## Development workflow

Features were built by a plan, build, review pipeline of Claude Code agents defined in `.claude/agents/`; progress is tracked in `specs/status.json` and each feature has a build brief in `specs/features/`. Real Voyager calls and the `.env` file are user-only: no agent reads secrets or calls the API, and tests use fakes and offline fixtures.

# GetWork

GetWork (CSE 573, Group 6) is a two-way resume and job matcher for software engineering and adjacent fields. Given a resume it returns the top-k job postings; given a posting it returns the top-k candidates. The full design builds a skill taxonomy from the corpus so that related skills (React and Vue) earn partial credit, then has an agent explain each match. See `docs/proposal.md` and `specs/architecture.html`.

## Status

| Stage | State |
| --- | --- |
| Resume preprocessing (filter, dedup) | Built |
| Posting preprocessing (domain filter, filter, dedup) | Built |
| Structurer (LLM extraction via ASU Voyager) | Built |
| Skill selection, clustering, re-labeling, document embeddings | Planned |
| Two-way retrieval and baselines, evaluation | Planned |
| FastAPI backend, React UI, explanation agent | Planned |

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
   [PLANNED] skill taxonomy (select, embed, cluster, re-label)
                                   |
                                   v
   [PLANNED] document embeddings -> two-way retrieval -> evaluation
                                   |
                                   v
   [PLANNED] FastAPI backend + React UI + explanation agent
```

## Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)
- git
- An ASU Voyager API key, only for real structurer runs (Linux or WSL). The Voyager portal, where you create the key, needs an ASURITE login and the ASU network or VPN. Tests and preprocessing do not need it.
- Raw data, which git ignores and you place yourself:
  - Resumes: `data/resume_corpus/resumes_corpus.zip` (the Jiechieu and Tsopze corpus).
  - Postings (LinkedIn Job Postings, Kaggle): under `data/job_postings/`, the preprocessor reads `companies/postings.csv`, `jobs/job_industries.csv` and `mappings/industries.csv`.

## Install and run

```
git clone https://github.com/oabuasba/CSE573-Group6-GetWork.git && cd CSE573-Group6-GetWork
uv sync
uv run pytest
```

Preprocess (each writes to `data/processed/<source>/` by default):

```
uv run python -m getwork.preprocess.run_resumes     # --zip PATH --out DIR
uv run python -m getwork.preprocess.run_postings    # --data DIR --out DIR
```

Structurer setup:

```
cp .env.example .env    # then set VOYAGER_API_KEY, VOYAGER_BASE_URL, VOYAGER_MODEL
uv run structurer --source resumes --preflight    # contacts Voyager (needs key, network and VOYAGER_MODEL); prints info for that model, exits 3 if not found
```

The full command list, exit codes and output files are in [`src/getwork/structure/README.md`](src/getwork/structure/README.md).

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

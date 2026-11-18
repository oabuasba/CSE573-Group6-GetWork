# Evaluation runs

Each folder is one `evaluate` run directory (`run.json`, `sets.json`, pools, blind pairs, judge
labels under `judges/`, and for finished runs `metrics.json`, `agreement.json`, `report.md`).
See `src/getwork/evaluation/README.md` for the commands.

## `preliminary_results/`

The run reported in the slides: artifacts `data/pipeline_run_v7`, two LLM judges (Gemma 4 31B
and Llama 4 Maverick 17B), rubric judge_v1 frozen before the test stage, 10 + 10 dev and
30 + 30 test queries, B1+ with default weights. Produced by
`scripts/bash/run_evaluation_preliminary.sh` (log: `v7_rubric_run.log`, not committed because of the `*.log` ignore rule). The deck reads
`metrics.json` and `agreement.json` from here.

## `other/`

Earlier and side runs, kept for comparison: `v6`, `v6.2` (older artifacts), `v7_smoke` and
`v7_smoke_v2` (10-pair smoke runs), `v7_heldout` (held-out dev pairs for rubric comparison).
Produced with `scripts/bash/run_evaluation.sh` or the preliminary script.

The `run.json` files record `artifacts_dir: artifacts_trial/v7`, the historical path of what is now `data/pipeline_run_v7`.

New runs default to `eval/run` (`--out-dir`).

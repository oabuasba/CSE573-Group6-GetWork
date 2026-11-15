# evaluation (feature: evaluation, part 1)

Measures whether the taxonomy-based system retrieves better matches than two baselines, on a
fixed random sample of queries over software and adjacent IT postings and the cleaned resumes.
It samples queries, pools the top results of the three systems, has two LLM judges label every
pooled resume-posting pair, and then computes nDCG@10, P@10, MRR, agreement and ARI with
bootstrap intervals and a randomization test. Everything except `evaluate judge` is offline.

## Commands

```
evaluate sample        [--seed N] [--pool-cap N] [--force]
evaluate pool          --stage dev|test [--k 10] [--weights S,T,L | --weights-file PATH] [--force]
evaluate judge         --preflight | --stage dev|test|tune [--dry-run] [--limit N] [--workers N]
evaluate freeze-rubric [--rubric-file PATH] [--force]
evaluate tune plan|pick [--step 0.1] [--k 10]
evaluate metrics       --labels judge|<author> --queries test|human [--allow-missing]
evaluate agreement     --scope judges|human [--stage dev|test] [--half all|first|second]
evaluate ari           [--k 10] [--seeds N]
evaluate report
```

Every command also takes `--out-dir` (default `eval_out`), `--artifacts-dir` and
`--processed-dir`. `bash scripts/run_evaluation.sh STAGE` wraps the run order below.

## What is measured

Three systems rank the same candidates: B1+ (skill and title Jaccard plus level; ADR-8), B2
(embedding of the flat skill list) and Full (embedding of the cluster-count document; ADR-1).
Pairs are judged blind: the judges never see which system retrieved a pair. The judges come
from different model families than the pipeline (ADR-9). Definitions of the metrics, the pool
and the human-check rule are in the definitions table of `specs/features/evaluation.html`.
The pool is described everywhere as "software and adjacent IT postings".

The existing retrieval README says `retrieve run` saves ranked lists "that evaluation pools and
judges". Evaluation does not use those files (ADR-37); it ranks again inside its own pool.

## Safety rules

- Outputs go only under `--out-dir` (default `eval_out`). Nothing is written under `data/` or
  `artifacts_trial/`; an `--out-dir` inside them is refused.
- `eval_out/` is not git-ignored on purpose: the authors' label files in `human/` cannot be
  recreated. Do not delete the folder; start a new `--out-dir` (such as `eval_out/v7`) instead.
- No agent runs the judge commands. Real Voyager judge runs are user-run only, and the script
  refuses the judge stages unless you set `USER_RUN_JUDGE=1` yourself.
- The `.env` file is read only by `evaluate judge`. The script and the other commands never read it.
- Re-sampling (`evaluate sample --force`) invalidates the judge, pair and label files of the
  old sample. Use a new `--out-dir` instead of re-sampling in place.

## Run order

One command runs everything (user-run, real Voyager calls; point `ARTIFACTS_DIR` at the pipeline
output):

```
USER_RUN_JUDGE=1 ARTIFACTS_DIR=artifacts_trial/v6 EVAL_DIR=eval_out/v6 bash scripts/run_evaluation.sh
```

With no argument the script runs `all`: prepare, dev-judge, tune-plan, tune-judge, tune-pick,
freeze, test-prepare, test-judge, metrics, report, and stops at the first failure. `smoke` is a
quick check (prepare, judge preflight, dev-judge limited to `JUDGE_LIMIT`, default 10). Both
refuse before any work unless `USER_RUN_JUDGE=1`. Any single stage name also still works.

`ARTIFACTS=artifacts_trial/v6` and `OUT=eval_out/v6` are written out in the commands below.

1. Sample and pool the dev stage (offline).

```
uv run evaluate sample --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate pool --stage dev --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
```

2. Judge the dev stage (user-run, real Voyager calls). Preflight first (EV-1), then count,
   then a 5-pair smoke run, then the full run. Read 20 to 30 rationales and edit the rubric
   until kappa is at least 0.6.

```
uv run evaluate judge --preflight --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate judge --stage dev --dry-run --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate judge --stage dev --limit 5 --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate judge --stage dev --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate agreement --scope judges --stage dev --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
```

3. Tune B1+ on the dev queries. `tune pick` must get the same `--step` and `--k` as
   `tune plan`, or the candidates differ and it refuses.

```
uv run evaluate tune plan --step 0.1 --k 10 --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate judge --stage tune --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate tune pick --step 0.1 --k 10 --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
```

4. Freeze the rubric, then pool and judge the test stage (user-run). Pooling the test stage
   also writes `pairs_blind_human.jsonl` for the authors.

```
uv run evaluate freeze-rubric --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate pool --stage test --weights-file eval_out/v6/tune/b1plus_weights.json --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate judge --stage test --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate agreement --scope judges --stage test --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
```

5. Human check: the two authors label `pairs_blind_human.jsonl` in the labeling UI (a later,
   separate feature). Their files appear under `human/` as `<author>.jsonl`.

6. Numbers (offline).

```
uv run evaluate metrics --labels judge --queries test --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate metrics --labels judge --queries human --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate metrics --labels <author> --queries human --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate agreement --scope human --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate ari --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
uv run evaluate report --artifacts-dir artifacts_trial/v6 --out-dir eval_out/v6
```

## The judges

The defaults are `gemma4-31b-it` (family `google`, judge-a) and `llama4-maverick-17b` (family
`meta`, judge-b). The user checked both in Voyager's `/v1/models`: context window 131,072 tokens
and strict-schema support. With `--reasoning-effort auto` (the default) gemma gets `none` and
llama4-maverick gets nothing, since it has no reasoning control. If a provider rejects the
strict schema, the judge falls back to plain JSON and the summary reports `schema_fallback`.

Independence rules: a judge must not be a model the pipeline uses. `qwen38-27b` (structuring,
relabeling, chat agent) is always checked; if the chat agent runs on another model, pass it
with `--also-used-model`. Overriding the judges needs `--judge-models` and `--judge-families`
together; one without the other is exit 2. Run `uv run evaluate judge --preflight` first.

## Cost control

`--dry-run` prints the pair count and the calls (2 x pairs) without a client. `--limit 5` is a
smoke run. `--workers` sets concurrency. A re-run retries only failed or missing pairs. Use
`tune plan` first (it prints the pair count) and a coarser `--step 0.5` if the count is large.

## The rubric loop

Edit `src/getwork/evaluation/prompts/judge_v1.md` (or pass `--rubric-file` with a copy such as
`judge_v2.md`), re-run dev, and check kappa. Then run `freeze-rubric`. Old rows for another
rubric hash are ignored, not deleted. The one allowed revision after the human check: delete
`rubric_frozen.json`, edit the rubric, freeze again, and re-judge the test stage.

## The human check

The authors label `pairs_blind_human.jsonl` (a subset of the test pairs). Each author's file is
`human/<author>.jsonl`, where the slug matches `[a-z0-9_-]{1,20}` and is not `judge`. Authors
must not open `judges/` or `labels_judge.parquet` before they finish labeling.

## Reading the results

- `metrics.json`: per system nDCG@10, P@10 and MRR with bootstrap intervals, gaps between
  systems and randomization-test p values (no multiple-comparison correction).
- `agreement.json`: Cohen kappa between judges, between authors, and judge against author, plus
  `fallback_decision`.
- `report.md`: the run, caveats (including the `b2_order` caveat) and all tables in one file.

## Manual checklist (user-run, real Voyager)

- EV-1: `evaluate judge --preflight` exits 0 and shows both judges with `supports_response_schema`
  true; `--stage dev --dry-run` prints counts; `--limit 5` gives 5 ok rows per judge.
- EV-2: the full dev run ends with 0 failed pairs; kappa at least 0.6 after the rubric loop.
- EV-3: `checked_against` lists `qwen38-27b` and the chat agent's model; judges are the defaults
  unless you overrode them on purpose.
- EV-4: after `freeze-rubric`, `judge --stage test` finishes with a small truncated count.
- EV-5: both authors finish `human/<author>.jsonl`.
- EV-6: `evaluate report` reads sensibly and lists the `b2_order` caveat.

## Not covered here

The labeling UI, the substitution test and the explanation evaluation belong to the other two
features (labeling UI in part 2; substitution and explanations in `evaluation-probes`).

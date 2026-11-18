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

Every command also takes `--out-dir` (default `eval/run`), `--artifacts-dir` and
`--processed-dir`. `bash scripts/bash/run_evaluation.sh STAGE` wraps the run order below.

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

- Outputs go only under `--out-dir` (default `eval/run`). Nothing is written under `data/` or
  `artifacts_trial/`; an `--out-dir` inside them is refused.
- `eval/` is not git-ignored on purpose: the authors' label files in `human/` cannot be
  recreated. Do not delete the folder; start a new `--out-dir` (such as `eval/other/v7`) instead.
- No agent runs the judge commands. Real Voyager judge runs are user-run only, and the script
  refuses the judge stages unless you set `USER_RUN_JUDGE=1` yourself.
- The `.env` file is read only by `evaluate judge`. The script and the other commands never read it.
- Re-sampling (`evaluate sample --force`) invalidates the judge, pair and label files of the
  old sample. Use a new `--out-dir` instead of re-sampling in place.

## Run order

One command runs everything (user-run, real Voyager calls; point `ARTIFACTS_DIR` at the pipeline
output):

```
USER_RUN_JUDGE=1 ARTIFACTS_DIR=artifacts_trial/v6 EVAL_DIR=eval/other/v6 bash scripts/bash/run_evaluation.sh
```

With no argument the script runs `all`: prepare, dev-judge, tune-plan, tune-judge, tune-pick,
freeze, test-prepare, test-judge, metrics, report, and stops at the first failure. `smoke` is a
quick check (prepare, judge preflight, dev-judge limited to `JUDGE_LIMIT`, default 10). Both
refuse before any work unless `USER_RUN_JUDGE=1`. Any single stage name also still works.
Both scripts print `[3/7] freeze` before each stage (a single stage shows `[1/1]`) and
`done: freeze (12s)` after it; a failed stage prints no `done:` line.

`ARTIFACTS=artifacts_trial/v6` and `OUT=eval/other/v6` are written out in the commands below.

1. Sample and pool the dev stage (offline).

```
uv run evaluate sample --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate pool --stage dev --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
```

2. Judge the dev stage (user-run, real Voyager calls). Preflight first (EV-1), then count,
   then a 5-pair smoke run, then the full run. Read 20 to 30 rationales and edit the rubric
   until kappa is at least 0.6.

```
uv run evaluate judge --preflight --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate judge --stage dev --dry-run --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate judge --stage dev --limit 5 --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate judge --stage dev --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate agreement --scope judges --stage dev --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
```

3. Tune B1+ on the dev queries. `tune pick` must get the same `--step` and `--k` as
   `tune plan`, or the candidates differ and it refuses.

```
uv run evaluate tune plan --step 0.1 --k 10 --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate judge --stage tune --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate tune pick --step 0.1 --k 10 --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
```

4. Freeze the rubric, then pool and judge the test stage (user-run). Pooling the test stage
   also writes `pairs_blind_human.jsonl` for the authors.

```
uv run evaluate freeze-rubric --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate pool --stage test --weights-file eval/other/v6/tune/b1plus_weights.json --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate judge --stage test --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate agreement --scope judges --stage test --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
```

5. Human check: the two authors label `pairs_blind_human.jsonl` in the labeling UI (a later,
   separate feature). Their files appear under `human/` as `<author>.jsonl`.

6. Numbers (offline).

```
uv run evaluate metrics --labels judge --queries test --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate metrics --labels judge --queries human --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate metrics --labels <author> --queries human --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate agreement --scope human --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate ari --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
uv run evaluate report --artifacts-dir artifacts_trial/v6 --out-dir eval/other/v6
```

## Preliminary run (no tuning, no human labels)

```
USER_RUN_JUDGE=1 ARTIFACTS_DIR=artifacts_trial/v6 EVAL_DIR=eval/other/v6_prelim bash scripts/bash/run_evaluation_preliminary.sh
```

`scripts/bash/run_evaluation_preliminary.sh` takes the same environment variables and safety gates as
`run_evaluation.sh` but skips the tune stages and the human check. Its stages are prepare,
dev-judge, freeze, test-prepare, test-judge, metrics, report (plus `smoke`). B1+ uses the default
weights (`run.json` records source `default`), the metrics stage runs `metrics --labels judge`
for the test and human query sets and `ari`, and it never runs `agreement --scope human`. Use a
fresh `EVAL_DIR`; sample and pool refuse to overwrite. `report.md` marks the run with the
caveats "B1+ untuned" and "Human check not run", so treat the numbers as preliminary.
Stage banners are `[i/7] name` (7 stages for `all`, 3 steps for `smoke`) with a `done:` line
and the elapsed seconds after each.

Optional variables of the preliminary script: `RUBRIC_FILE` (passed as `--rubric-file` to the
judge stages, to `agreement` and to `freeze-rubric`, and checked up front), and `TEST_RESUMES` and
`TEST_POSTINGS` (passed to `sample` only). The test sizes must be integers of at least 10: the
human sets (10 + 10 by default) are drawn from the test sets, and `sample` otherwise fails with
"cannot draw 10 ... from only N" (exit 2), so the script exits 2 before any work. The dev sets do
not depend on the test sizes (each set has its own seeded draw), so the dev pairs stay the same.

## Rubric comparison (judge_v1 against judge_v2, preliminary)

`prompts/judge_v2.md` keeps the output contract of `judge_v1.md` and adds a checklist for label 2,
the rule that an unmet or unmentioned mandatory requirement caps the label at 1 (a preferred item
does not), and four short examples. `evaluate compare-rubrics` (offline, read only) groups the
judge rows of one or more `--out-dir` folders by rubric hash and, on the pairs judged by both
judges under every compared rubric, prints per rubric: n, exact agreement, Cohen kappa (3 classes),
quadratic-weighted kappa, kappa of "label 2 vs not", kappa of ">=1 vs 0", the judge-a=1/judge-b=2
and reverse disagreement counts (total and per direction) and each judge's label distribution.
`--rubric-hash` (repeatable) selects the rubrics (two or more; all found by default); the output also gives `overlap_pairs`. `--format table` prints one summary line per rubric instead of JSON. It reuses
`agreement.agreement_stats` and `stats.cohen_kappa`: standard Cohen kappa over the fixed classes
0, 1, 2 (the quadratic version scales the loss by (i-j)^2/(k-1)^2), except that it returns null
(not NaN, not 1) when the chance-expected disagreement is zero, for example when both judges used
one single label.

Runbook (user-run; about 400 judge calls, 100 pairs x 2 judges x 2 rubrics, roughly 15 minutes at
4 workers: judge-a about 3.4 s per pair and judge-b about 0.9 s per pair, so about 6 plus 2
minutes per rubric). Use a new folder; `eval/other/v7_smoke` is only read.

1. Offline prepare into the new folder (same default seed 0 as the smoke run, so the dev pairs are
   the same; their ids are `sha256(seed:resume:posting)` and `judge --limit N` takes the first N
   by sorted pair id, so the 10 smoke pairs, which are the first 10 sorted dev pair ids of the 432,
   are inside the first 100 as long as the dev pair set is unchanged). `TEST_*` is optional and
   only makes the later test stage cheaper:

```
ARTIFACTS_DIR=data/pipeline_run_v7 EVAL_DIR=eval/preliminary_results TEST_RESUMES=30 TEST_POSTINGS=30 bash scripts/bash/run_evaluation_preliminary.sh prepare
```

   Optional, saves 20 calls: copy the smoke judge rows (rubric hash judge_v1) so those pairs are
   skipped; rows of the same hash and pair are de-duplicated by the compare step.

```
mkdir -p eval/preliminary_results/judges && cp -n eval/other/v7_smoke/judges/*.jsonl eval/preliminary_results/judges/
```

2. Judge the first 100 dev pairs under each rubric (user-run). Rows of different rubric hashes
   coexist in the same `judges/judge-*.jsonl`: a run appends and only skips pairs already ok under
   its own hash, it never rewrites other rows (tested). Each run also rebuilds
   `labels_judge.parquet` from its own hash only.

```
uv run evaluate judge --preflight --artifacts-dir data/pipeline_run_v7 --out-dir eval/preliminary_results
uv run evaluate judge --stage dev --limit 100 --rubric-file src/getwork/evaluation/prompts/judge_v2.md --artifacts-dir data/pipeline_run_v7 --out-dir eval/preliminary_results
uv run evaluate judge --stage dev --limit 100 --artifacts-dir data/pipeline_run_v7 --out-dir eval/preliminary_results
```

3. Compare (offline). The hashes are in the output; the baseline pairs of `v7_smoke` are included
   for free, and the overlap is the 100 pairs judged under both rubrics.

```
uv run evaluate compare-rubrics --out-dir eval/other/v7_smoke --out-dir eval/preliminary_results
```

4. Freeze the winner and continue (stage by stage, because `prepare` refuses to overwrite). Set
   `RUBRIC_FILE` on every stage (leave it unset for judge_v1). The test judge refuses a rubric that
   differs from the frozen one, and `agreement` before the freeze uses the rubric you pass, not
   the frozen one. `JUDGE_LIMIT=100` on dev-judge makes both judges skip the 100 pairs they
   already labeled under that rubric (no second payment) and just rebuild the labels and the dev
   agreement; without it the stage would judge all 432 dev pairs. Do not set `JUDGE_LIMIT` for
   test-judge.

```
export RUBRIC_FILE=src/getwork/evaluation/prompts/judge_v2.md ARTIFACTS_DIR=data/pipeline_run_v7 EVAL_DIR=eval/preliminary_results
USER_RUN_JUDGE=1 JUDGE_LIMIT=100 bash scripts/bash/run_evaluation_preliminary.sh dev-judge
bash scripts/bash/run_evaluation_preliminary.sh freeze
bash scripts/bash/run_evaluation_preliminary.sh test-prepare
USER_RUN_JUDGE=1 bash scripts/bash/run_evaluation_preliminary.sh test-judge
bash scripts/bash/run_evaluation_preliminary.sh metrics
bash scripts/bash/run_evaluation_preliminary.sh report
```

   The dev-judge stage runs `judge --preflight` first (a model-list request, no judging). The
   test stage is new work either way (with `TEST_*=30` that is 2 x pairs of 30 + 30 queries).
   The frozen rubric is one hash, so `labels_judge.parquet` always comes from the rubric of the
   last `judge` run; if you froze v1 instead, run step 4 without `RUBRIC_FILE`.

### judge_v3 on the same 100 pairs

`prompts/judge_v3.md` (hash `sha256:b71eaf2d8bd1`) separates credentials (the only items that cap
the label at 1) from core skills (judged by the label rules), asks for evidence in the resume
text, and states the policy to ignore location and work-authorization conditions and to judge
against the best-matching role of a multi-role posting. `eval/preliminary_results` already holds the v1
(`sha256:87ecca1b14a7`) and v2 (`sha256:9e331cf31118`) rows of the first 100 dev pairs, so only
v3 is new: about 200 judge calls (100 pairs x 2 judges), roughly 6 minutes for judge-a and 2
minutes for judge-b at 4 workers. The preflight was done in the first run and is not needed
again. Run it with the same `--limit 100`, then compare the three on the 100 pairs judged under
all of them (`--format table` prints one line per rubric, the default prints JSON):

```
uv run evaluate judge --stage dev --limit 100 --rubric-file src/getwork/evaluation/prompts/judge_v3.md --artifacts-dir data/pipeline_run_v7 --out-dir eval/preliminary_results
uv run evaluate compare-rubrics --out-dir eval/other/v7_smoke --out-dir eval/preliminary_results --format table
uv run evaluate compare-rubrics --out-dir eval/other/v7_smoke --out-dir eval/preliminary_results
```

To freeze and continue, use step 4 above with `RUBRIC_FILE` set to the winner (`judge_v3.md`,
`judge_v2.md`, or unset for v1); `freeze-rubric --rubric-file PATH` is what the `freeze` stage runs.
`JUDGE_LIMIT=100` on dev-judge skips the 100 pairs already judged under the chosen rubric.

Warning: a rubric edited while reading the disagreements of these 100 pairs can look better on
them than it is on new pairs (the change is fitted to them). Optional held-out check, on dev pairs
101 to 200 by sorted pair id (`--offset 100 --limit 100`; `--offset N`, default 0, skips the first
N pairs of the sorted list and is combined with `--limit`). Use a separate folder, so the compare
overlap holds only the new pairs (in `v7_rubric` the old and new rubric overlap would include the
first 100 again). The same seed gives the same dev pairs; judge the old rubric (v1, default) and
the new one on the slice and compare (about 400 calls):

```
ARTIFACTS_DIR=data/pipeline_run_v7 EVAL_DIR=eval/other/v7_heldout bash scripts/bash/run_evaluation_preliminary.sh prepare
uv run evaluate judge --preflight --artifacts-dir data/pipeline_run_v7 --out-dir eval/other/v7_heldout
uv run evaluate judge --stage dev --offset 100 --limit 100 --artifacts-dir data/pipeline_run_v7 --out-dir eval/other/v7_heldout
uv run evaluate judge --stage dev --offset 100 --limit 100 --rubric-file src/getwork/evaluation/prompts/judge_v3.md --artifacts-dir data/pipeline_run_v7 --out-dir eval/other/v7_heldout
uv run evaluate compare-rubrics --out-dir eval/other/v7_heldout --format table
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
smoke run. `--workers` sets concurrency. `--progress-every N` (default 25, 0 disables) prints
`[judge-a] judged 50/200 pairs (48 ok, 2 failed, 0 skipped as already done) elapsed 61s, ETA 183s`
to stderr every N judged pairs and at the end of each judge; the total includes pairs already
labeled by an earlier run, and the lines carry counts and times only. A re-run retries only failed or missing pairs. Use
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

#!/usr/bin/env bash
# Runs a PRELIMINARY evaluation: no B1+ tuning and no human labels, still ending in metrics and a
# report (see src/getwork/evaluation/README.md). Use run_evaluation.sh for the full evaluation.
# Usage: [VAR=...] bash scripts/bash/run_evaluation_preliminary.sh [STAGE]
#   STAGE: all (default when omitted), smoke, or one of
#          prepare dev-judge freeze test-prepare test-judge metrics report
#   all     runs every single stage above in that order and stops at the first failure
#   smoke   quick check: prepare, judge --preflight, then dev-judge limited to JUDGE_LIMIT (default 10)
# Example (needs a fresh EVAL_DIR, because sample and pool refuse to overwrite earlier outputs):
#   USER_RUN_JUDGE=1 ARTIFACTS_DIR=artifacts_trial/v6 EVAL_DIR=eval/other/v6_prelim \
#     bash scripts/bash/run_evaluation_preliminary.sh
#   ARTIFACTS_DIR   retrieval artifacts to evaluate, read only (default artifacts_trial)
#   PROCESSED_DIR   preprocessed parquet, read only (default data/processed)
#   EVAL_DIR        the only output folder (default eval/run); use a fresh one for each run
#   JUDGE_MODELS, JUDGE_FAMILIES  optional; passed only when both are set; one alone exits 2
#   ALSO_USED_MODELS  optional, space-separated models the judges must also differ from
#   JUDGE_LIMIT     optional, adds --limit N to the judge calls (smoke defaults it to 10)
#   WORKERS         concurrent judge calls (default 4)
#   RUBRIC_FILE     optional rubric to judge with, for example
#                   src/getwork/evaluation/prompts/judge_v2.md; passed as --rubric-file to the judge
#                   stages, to agreement and to freeze-rubric (default: the packaged judge_v1.md).
#                   Set it on every stage of one run: agreement before the freeze, and the test
#                   judge, check the rubric hash against it or against the frozen one.
#   TEST_RESUMES, TEST_POSTINGS  optional sizes of the test query sets (the sample default is 100
#                   each), passed to sample only; positive integers of at least 10, the size of the
#                   human sets drawn from the test sets, else exit 2 before any work
# Caveats of a preliminary run:
#   - B1+ is untuned: the test stage uses the default weights (run.json records source "default").
#   - There is no human check: no agreement --scope human and no per-author metrics; the judge
#     labels are unvalidated against people.
#   - The numbers are preliminary; report.md says so in its caveats.
# The judge stages (dev-judge, test-judge) make real Voyager calls and are run by the user only:
# they refuse to start unless USER_RUN_JUDGE=1 is set; all and smoke refuse up front, before any
# work. Every command is printed as it runs. Every other stage is offline.
# This script never passes --force.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

STAGES="prepare dev-judge freeze test-prepare test-judge metrics report"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts_trial}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
EVAL_DIR="${EVAL_DIR:-eval/run}"
WORKERS="${WORKERS:-4}"
MIN_TEST_SIZE=10  # the human sets (10 + 10 by default) are drawn from the test sets

GROUPS_LIST="all smoke"

usage() {
  echo "usage: bash scripts/bash/run_evaluation_preliminary.sh [{all|smoke|${STAGES// /|}}]" >&2
  exit 2
}

step() { echo "==> $*"; "$@"; }

# numbered NAME CMD...: prints "[i/TOTAL] NAME", runs CMD, then "done: NAME (Ns)" (not on failure).
STAGE_INDEX=0
STAGE_TOTAL=1
numbered() {
  local name="$1" started=$SECONDS
  shift
  STAGE_INDEX=$((STAGE_INDEX + 1))
  echo "[$STAGE_INDEX/$STAGE_TOTAL] $name"
  "$@"
  echo "done: $name ($((SECONDS - started))s)"
}

[[ $# -le 1 ]] || usage
STAGE="${1:-all}"
[[ " $GROUPS_LIST $STAGES " == *" $STAGE "* ]] || usage
if [[ "$STAGE" == "smoke" ]]; then
  JUDGE_LIMIT="${JUDGE_LIMIT:-10}"
fi

if [[ -n "${JUDGE_MODELS:-}" || -n "${JUDGE_FAMILIES:-}" ]]; then
  if [[ -z "${JUDGE_MODELS:-}" || -z "${JUDGE_FAMILIES:-}" ]]; then
    echo "error: JUDGE_MODELS and JUDGE_FAMILIES must be set together" >&2
    exit 2
  fi
fi

require_test_size() {
  local name="$1" value="${!1:-}"
  [[ -z "$value" ]] && return 0
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]] || (( value < MIN_TEST_SIZE )); then
    echo "error: $name must be a positive integer of at least $MIN_TEST_SIZE" \
         "(the human sets are drawn from the test sets), got '$value'" >&2
    exit 2
  fi
}
require_test_size TEST_RESUMES
require_test_size TEST_POSTINGS
if [[ -n "${RUBRIC_FILE:-}" && ! -f "$RUBRIC_FILE" ]]; then
  echo "error: RUBRIC_FILE not found: $RUBRIC_FILE" >&2
  exit 2
fi

BASE=(--artifacts-dir "$ARTIFACTS_DIR" --processed-dir "$PROCESSED_DIR" --out-dir "$EVAL_DIR")
JUDGE_ARGS=("${BASE[@]}" --workers "$WORKERS")
if [[ -n "${JUDGE_MODELS:-}" ]]; then
  JUDGE_ARGS+=(--judge-models "$JUDGE_MODELS" --judge-families "$JUDGE_FAMILIES")
fi
for model in ${ALSO_USED_MODELS:-}; do
  JUDGE_ARGS+=(--also-used-model "$model")
done
if [[ -n "${JUDGE_LIMIT:-}" ]]; then
  JUDGE_ARGS+=(--limit "$JUDGE_LIMIT")
fi
RUBRIC_ARGS=()
if [[ -n "${RUBRIC_FILE:-}" ]]; then
  RUBRIC_ARGS=(--rubric-file "$RUBRIC_FILE")
  JUDGE_ARGS+=("${RUBRIC_ARGS[@]}")
fi
SAMPLE_ARGS=()
if [[ -n "${TEST_RESUMES:-}" ]]; then
  SAMPLE_ARGS+=(--test-resumes "$TEST_RESUMES")
fi
if [[ -n "${TEST_POSTINGS:-}" ]]; then
  SAMPLE_ARGS+=(--test-postings "$TEST_POSTINGS")
fi

evaluate() { step uv run evaluate "$@"; }

require_user_run() {
  if [[ "${USER_RUN_JUDGE:-}" != "1" ]]; then
    echo "error: stage $STAGE makes real Voyager calls and is user-run only;" >&2
    echo "       set USER_RUN_JUDGE=1 yourself to run it (try --dry-run by hand first)" >&2
    exit 2
  fi
}

# No agreement --scope human and no per-author metrics: there are no human label files.
run_metrics() {
  evaluate metrics "${BASE[@]}" --labels judge --queries test
  evaluate metrics "${BASE[@]}" --labels judge --queries human
  evaluate ari "${BASE[@]}"
}

run_stage() {
  case "$1" in
    prepare)
      evaluate sample "${BASE[@]}" ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"}
      evaluate pool "${BASE[@]}" --stage dev
      ;;
    dev-judge)
      require_user_run
      evaluate judge "${JUDGE_ARGS[@]}" --preflight
      evaluate judge "${JUDGE_ARGS[@]}" --stage dev
      evaluate agreement "${BASE[@]}" ${RUBRIC_ARGS[@]+"${RUBRIC_ARGS[@]}"} --scope judges --stage dev
      ;;
    freeze)
      evaluate freeze-rubric "${BASE[@]}" ${RUBRIC_ARGS[@]+"${RUBRIC_ARGS[@]}"}
      ;;
    test-prepare)
      evaluate pool "${BASE[@]}" --stage test
      ;;
    test-judge)
      require_user_run
      evaluate judge "${JUDGE_ARGS[@]}" --stage test
      evaluate agreement "${BASE[@]}" ${RUBRIC_ARGS[@]+"${RUBRIC_ARGS[@]}"} --scope judges --stage test
      ;;
    metrics)
      run_metrics
      ;;
    report)
      evaluate report "${BASE[@]}"
      ;;
  esac
}

case "$STAGE" in
  all)
    require_user_run
    read -r -a ALL_STAGES <<< "$STAGES"
    STAGE_TOTAL=${#ALL_STAGES[@]}
    for stage in "${ALL_STAGES[@]}"; do
      numbered "$stage" run_stage "$stage"
    done
    ;;
  smoke)
    require_user_run
    STAGE_TOTAL=3
    numbered prepare run_stage prepare
    numbered preflight evaluate judge "${JUDGE_ARGS[@]}" --preflight
    numbered dev-judge evaluate judge "${JUDGE_ARGS[@]}" --stage dev
    ;;
  *)
    numbered "$STAGE" run_stage "$STAGE"
    ;;
esac

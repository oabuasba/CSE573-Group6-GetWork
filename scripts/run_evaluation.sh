#!/usr/bin/env bash
# Runs the evaluation harness end to end, or one stage of it (see src/getwork/evaluation/README.md).
# Usage: [VAR=...] bash scripts/run_evaluation.sh [STAGE]
#   STAGE: all (default when omitted), smoke, or one of
#          prepare dev-judge tune-plan tune-judge tune-pick freeze test-prepare test-judge metrics report
#   all     runs every single stage above in that order and stops at the first failure
#   smoke   quick check: prepare, judge --preflight, then dev-judge limited to JUDGE_LIMIT (default 10);
#           no tune, freeze or test stages
# Point ARTIFACTS_DIR at the pipeline's output folder, for example:
#   USER_RUN_JUDGE=1 ARTIFACTS_DIR=artifacts_trial/v6 EVAL_DIR=eval_out/v6 bash scripts/run_evaluation.sh
#   ARTIFACTS_DIR   retrieval artifacts to evaluate, read only (default artifacts_trial)
#   PROCESSED_DIR   preprocessed parquet, read only (default data/processed)
#   EVAL_DIR        the only output folder (default eval_out)
#   JUDGE_MODELS, JUDGE_FAMILIES  optional; passed only when both are set (else the command defaults
#                   apply: gemma4-31b-it/google and llama4-maverick-17b/meta); one alone exits 2
#   ALSO_USED_MODELS  optional, space-separated models the judges must also differ from
#   JUDGE_LIMIT     optional, adds --limit N to the judge calls (smoke defaults it to 10)
#   WORKERS         concurrent judge calls (default 4)
# The judge stages (dev-judge, tune-judge, test-judge) make real Voyager calls and are run by the
# user only: they refuse to start unless USER_RUN_JUDGE=1 is set; all and smoke refuse up front,
# before any work. Every command is printed as it runs.
# Every other stage is offline. This script never passes --force.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGES="prepare dev-judge tune-plan tune-judge tune-pick freeze test-prepare test-judge metrics report"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts_trial}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
EVAL_DIR="${EVAL_DIR:-eval_out}"
WORKERS="${WORKERS:-4}"

GROUPS_LIST="all smoke"

usage() { echo "usage: bash scripts/run_evaluation.sh [{all|smoke|${STAGES// /|}}]" >&2; exit 2; }

step() { echo "==> $*"; "$@"; }

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

evaluate() { step uv run evaluate "$@"; }

require_user_run() {
  if [[ "${USER_RUN_JUDGE:-}" != "1" ]]; then
    echo "error: stage $STAGE makes real Voyager calls and is user-run only;" >&2
    echo "       set USER_RUN_JUDGE=1 yourself to run it (try --dry-run by hand first)" >&2
    exit 2
  fi
}

run_metrics() {
  evaluate metrics "${BASE[@]}" --labels judge --queries test
  evaluate metrics "${BASE[@]}" --labels judge --queries human
  local file author
  for file in "$EVAL_DIR"/human/*.jsonl; do
    [[ -e "$file" ]] || continue
    author="$(basename "$file" .jsonl)"
    evaluate metrics "${BASE[@]}" --labels "$author" --queries human
  done
  evaluate agreement "${BASE[@]}" --scope human
  evaluate ari "${BASE[@]}"
}

run_stage() {
  case "$1" in
    prepare)
      evaluate sample "${BASE[@]}"
      evaluate pool "${BASE[@]}" --stage dev
      ;;
    dev-judge)
      require_user_run
      evaluate judge "${JUDGE_ARGS[@]}" --preflight
      evaluate judge "${JUDGE_ARGS[@]}" --stage dev
      evaluate agreement "${BASE[@]}" --scope judges --stage dev
      ;;
    tune-plan)
      evaluate tune plan "${BASE[@]}"
      ;;
    tune-judge)
      require_user_run
      evaluate judge "${JUDGE_ARGS[@]}" --stage tune
      ;;
    tune-pick)
      evaluate tune pick "${BASE[@]}"
      ;;
    freeze)
      evaluate freeze-rubric "${BASE[@]}"
      ;;
    test-prepare)
      evaluate pool "${BASE[@]}" --stage test --weights-file "$EVAL_DIR/tune/b1plus_weights.json"
      ;;
    test-judge)
      require_user_run
      evaluate judge "${JUDGE_ARGS[@]}" --stage test
      evaluate agreement "${BASE[@]}" --scope judges --stage test
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
    for stage in $STAGES; do
      run_stage "$stage"
    done
    ;;
  smoke)
    require_user_run
    run_stage prepare
    evaluate judge "${JUDGE_ARGS[@]}" --preflight
    evaluate judge "${JUDGE_ARGS[@]}" --stage dev
    ;;
  *)
    run_stage "$STAGE"
    ;;
esac

#!/usr/bin/env bash
# Runs every pipeline step between structuring and the web client.
# Usage: [STRUCTURED_DIR=...] [ARTIFACTS_DIR=...] [PROCESSED_DIR=...] bash scripts/run_pipeline.sh [CONFIG_ID]
#   STRUCTURED_DIR  structurer output, read by alias-table only (default data/structured)
#   ARTIFACTS_DIR   output tree for every later step (default artifacts)
#   PROCESSED_DIR   preprocessed parquet, read by relabel and doc-embed build (default data/processed)
# `taxonomy select` always runs right after grid, so the cluster count is fixed: the configuration is CONFIG_ID
#   (first argument), else $CLUSTER_CONFIG_ID, else kmeans-01 (k=10, exactly 10 clusters, no noise). Grid's own pick is overridden.
#   CLUSTER_CONFIG_ID  configuration id for `taxonomy select` when no argument is given (default kmeans-01)
#   NAME_METHOD     how `taxonomy name` writes cluster names: llm (default, real LLM calls) or heuristic
#                   (offline runs only; member-list names then need --allow-member-names at build)
#   NAME_OVERLAP_ROUNDS  overlap review rounds after naming (default 2; 0 disables it; each round can make real LLM calls when NAME_METHOD is llm)
#   NAME_WORKERS    concurrent naming calls for `taxonomy name` (default 4; real LLM calls when NAME_METHOD is llm)
# `taxonomy plot` (after build) writes the cluster pictures to $ARTIFACTS_DIR/taxonomy/plots/ (offline, no model calls).
# The embed steps make real Voyager calls. `name --force` overwrites cluster_names.csv, and grid/select
# overwrite selection.json, so existing copies are first archived to $ARTIFACTS_DIR/taxonomy/archive/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STRUCTURED_DIR="${STRUCTURED_DIR:-data/structured}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
NAME_METHOD="${NAME_METHOD:-llm}"
NAME_WORKERS="${NAME_WORKERS:-4}"
NAME_OVERLAP_ROUNDS="${NAME_OVERLAP_ROUNDS:-2}"
CLUSTER_CONFIG_ID="${1:-${CLUSTER_CONFIG_ID:-kmeans-01}}"
TAXONOMY_DIR="$ARTIFACTS_DIR/taxonomy"

step() { echo "==> $*"; "$@"; }

archive() {  # archive FILE: copy FILE to taxonomy/archive/<stem>_<UTC timestamp>.<ext>; quiet if absent
  local file="$1"
  [[ -f "$file" ]] || return 0
  local name
  name="$(basename "$file")"
  mkdir -p "$TAXONOMY_DIR/archive"
  cp "$file" "$TAXONOMY_DIR/archive/${name%.*}_$(date -u +%Y%m%dT%H%M%S%NZ).${name##*.}"
}

EMBED=(--embed-backend voyager --embed-model qwen3-embedding-8b)
ART=(--artifacts-dir "$ARTIFACTS_DIR")
BUILD_ARGS=()
if [[ "$NAME_METHOD" == "heuristic" ]]; then
  BUILD_ARGS=(--allow-member-names)
fi

step uv run alias-table build --structured-dir "$STRUCTURED_DIR" "${ART[@]}"
step uv run alias-table apply --structured-dir "$STRUCTURED_DIR" "${ART[@]}"
step uv run taxonomy counts "${ART[@]}"
step uv run taxonomy embed "${ART[@]}" "${EMBED[@]}"
archive "$TAXONOMY_DIR/selection.json"
step uv run taxonomy grid "${ART[@]}"
step uv run taxonomy select "${ART[@]}" --config-id "$CLUSTER_CONFIG_ID"
archive "$TAXONOMY_DIR/cluster_names.csv"
step uv run taxonomy name "${ART[@]}" --method "$NAME_METHOD" --workers "$NAME_WORKERS" --overlap-rounds "$NAME_OVERLAP_ROUNDS" --force
step uv run taxonomy build "${ART[@]}" "${BUILD_ARGS[@]}"
step uv run taxonomy plot "${ART[@]}"
step uv run relabel "${ART[@]}" --processed-dir "$PROCESSED_DIR"
step uv run doc-embed build "${ART[@]}" --processed-dir "$PROCESSED_DIR"
step uv run doc-embed embed "${ART[@]}" "${EMBED[@]}"
step uv run retrieve run "${ART[@]}"

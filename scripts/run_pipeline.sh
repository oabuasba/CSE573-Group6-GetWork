#!/usr/bin/env bash
# Runs every pipeline step between structuring and the web client.
# Usage: [STRUCTURED_DIR=...] [ARTIFACTS_DIR=...] [PROCESSED_DIR=...] bash scripts/run_pipeline.sh [CONFIG_ID]
#   STRUCTURED_DIR  structurer output, read by alias-table only (default data/structured)
#   ARTIFACTS_DIR   output tree for every later step (default artifacts)
#   PROCESSED_DIR   preprocessed parquet, read by relabel and doc-embed build (default data/processed)
# With CONFIG_ID (e.g. kmeans-04), `taxonomy select` runs right after grid; otherwise grid's own pick in selection.json is used (the previous one is archived).
#   NAME_METHOD     how `taxonomy name` writes cluster names: llm (default, real LLM calls) or heuristic
#                   (offline runs only; member-list names then need --allow-member-names at build)
#   NAME_WORKERS    concurrent naming calls for `taxonomy name` (default 4; real LLM calls when NAME_METHOD is llm)
# The embed steps make real Voyager calls. `name --force` overwrites cluster_names.csv, and grid/select
# overwrite selection.json, so existing copies are first archived to $ARTIFACTS_DIR/taxonomy/archive/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STRUCTURED_DIR="${STRUCTURED_DIR:-data/structured}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
NAME_METHOD="${NAME_METHOD:-llm}"
NAME_WORKERS="${NAME_WORKERS:-4}"
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
if [[ -n "${1:-}" ]]; then
  step uv run taxonomy select "${ART[@]}" --config-id "$1"
fi
archive "$TAXONOMY_DIR/cluster_names.csv"
step uv run taxonomy name "${ART[@]}" --method "$NAME_METHOD" --workers "$NAME_WORKERS" --force
step uv run taxonomy build "${ART[@]}" "${BUILD_ARGS[@]}"
step uv run relabel "${ART[@]}" --processed-dir "$PROCESSED_DIR"
step uv run doc-embed build "${ART[@]}" --processed-dir "$PROCESSED_DIR"
step uv run doc-embed embed "${ART[@]}" "${EMBED[@]}"
step uv run retrieve run "${ART[@]}"

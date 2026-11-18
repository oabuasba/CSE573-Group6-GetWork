#!/usr/bin/env bash
# Shows top-K postings for a random (or given) resume under each retrieval system.
# Usage: [ARTIFACTS_DIR=...] [K=5] bash scripts/bash/match_random.sh [RESUME_ID]
# Read-only: `retrieve show` uses stored vectors, so no Voyager/LLM calls are made.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
K="${K:-5}"
RESUMES="$ARTIFACTS_DIR/documents/resumes/resumes_documents.parquet"

if [[ ! -f "$RESUMES" ]]; then
  echo "error: $RESUMES not found; run scripts/bash/run_pipeline.sh first" >&2
  exit 1
fi

if [[ $# -ge 1 ]]; then
  ID="$1"
else
  ID="$(uv run python -c 'import sys, pandas as pd; print(pd.read_parquet(sys.argv[1], columns=["id"])["id"].sample(1).iloc[0])' "$RESUMES")"
fi

echo "Resume id: $ID"
for SYSTEM in B1+ B2 Full; do
  echo
  echo "===== System: $SYSTEM (top $K postings) ====="
  uv run retrieve show --artifacts-dir "$ARTIFACTS_DIR" --direction resume_to_posting --id "$ID" --system "$SYSTEM" --k "$K"
done

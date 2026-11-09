# retrieval (pipeline step 7)

Ranks postings for every resume and resumes for every posting with three systems (B1+, B2,
Full), exactly and reproducibly, and saves the ranked lists that evaluation pools and judges.
No key, model or GPU is needed: B2 and Full multiply the stored document vectors, and B1+ needs
only the documents tables.

## Systems

- **Full** and **B2**: score = inner product of unit vectors (`S = R J^T`), computed in blocks of
  queries so the whole matrix is never in memory. Vectors come from `doc-embed embed`.
- **B1+**: `score = w_s * J_skills + w_t * J_title + w_l * L`.
  `J_skills` is the Jaccard index of the two final skill sets (compared by `normalize_key`),
  `J_title` the Jaccard index of lower-cased title tokens without stop words, and
  `L = 1 - |rank_r - rank_j| / 5` over the six experience levels (0.5 when a level is
  unspecified). Default weights `0.7,0.15,0.15`. The formula and defaults answer Q-2 as a
  proposal; confirm or change them before the dev stage (only `b1plus.py` changes).

## Ties

Per query, documents are ordered by score descending and, for equal scores, by document
position ascending (a stable sort of the negated scores). `--k` larger than the number of
documents is clipped; the summary reports the smallest effective k as `k_effective`.

## Commands

```
retrieve run  [--artifacts-dir artifacts_trial] [--systems all|B1+,B2,Full] [--k 10]
              [--weights 0.7,0.15,0.15 | --weights-file PATH] [--block-size 512]
retrieve show --direction resume_to_posting|posting_to_resume --id ID [--system B1+|B2|Full]
              [--k 5] [--artifacts-dir artifacts_trial] [--weights ... | --weights-file PATH]
```

`run` checks every input before writing anything: both documents tables and summaries must
exist and be comparable (same taxonomy, relabel mode, levels and serialization), and for B2
and Full the vectors must match the current texts, the table ids and each other (same
embedding model). Exit code 0 is success, 2 is a usage, config or input error. `show` computes
one query from the stored inputs (it does not read the Parquet files) and prints the query's
title, level and skills, then one line per match: rank, score, id, level, title, shared skills.

## Outputs

Under `<artifacts-dir>/retrieval/`:

- `topk_{b1plus,b2,full}_{resume_to_posting,posting_to_resume}.parquet`: columns `system`
  (B1+, B2 or Full), `direction`, `query_id`, `rank` (1-based int32), `doc_id`, `score`
  (float32); rows ordered by query position then rank.
- `retrieval_summary.json`: `k`, `k_effective`, `systems`, counts, `weights` (null without
  B1+), `embedding` (null without B2 or Full), `provenance` and the row count per file.

## Applying tuned weights

Tuning needs judge labels and a metric, so it happens in the evaluation stages with
`grid_search_weights`. To apply the result, write a JSON file and pass it with
`--weights-file`:

```json
{"skills": 0.6, "title": 0.2, "level": 0.2}
```

The three weights must be non-negative and sum to 1. The summary records them.

## User-run checklist

### RV-1 B1+ only (needs only the documents tables)

`uv run retrieve run --systems B1+`. Pass: exit 0, two B1+ Parquet files exist, the summary
shows the weights.

### RV-2 All three systems (after doc-embed embed)

`uv run retrieve run`. Pass: exit 0, six Parquet files, the summary names the embedding model
and the provenance values.

### RV-3 Inspect matches

`uv run retrieve show --direction resume_to_posting --id <a resume id> --system Full`, then
`--system B1+`, then `--direction posting_to_resume --id <a posting id>`. Pass: the matches look
plausible against the query's skills and title (with 100 + 100 trial records expect rough but
sensible neighbors, not clean results).

### RV-4 Check the working tree

`git status`. Pass: nothing under `artifacts_trial/`, `data/structured/` or `data/processed/`
appears as changed.

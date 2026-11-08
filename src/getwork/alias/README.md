# Alias table runbook

The structurer stores skills exactly as the model wrote them, so one skill shows up as `JAVA`,
`Java`, `java`, or `JS` and `JavaScript`. `alias-table` merges those variants into one canonical
spelling per skill and rewrites every structured record through the table. Everything downstream
(counting, clustering, lookup, comparison) reads the canonical records, not the structured ones.

The table has two layers. First, every spelling in the corpus is grouped by a normalized key
(case-folded, whitespace collapsed, trailing punctuation removed) and the group takes its best
spelling. Second, the curated file `src/getwork/alias/alias_curated.csv` (`alias,canonical` rows)
overrides it for real abbreviations (`js` becomes `JavaScript`). Conflicting curated rows fail the
build with the offending row named.

## Commands

All commands default to the trial directories: `--structured-dir data/structured_trial` and
`--artifacts-dir artifacts_trial` (git-ignored). Nothing under `data/structured/` or
`data/processed/` is ever written.

```
uv run alias-table build     # structured records -> artifacts_trial/alias/
uv run alias-table apply     # table + structured records -> artifacts_trial/canonical/<source>/
uv run alias-table propose --model <id>   # optional, uses your key or local server
```

`alias-table build` takes `--curated PATH` to use another curated CSV. `alias-table propose`
takes the same `--backend`, `--model`, `--env-file`, and `--extra-body` options as the structurer,
plus `--batch-size`, `--reference-size`, `--max-skills`, and `--max-tokens`.

Exit codes: 0 ok, 1 propose finished with failed batches, 2 usage or config error (missing input,
missing table, corrupt record, bad curated file), 3 fatal LLM error.

## Output files

- `artifacts_trial/alias/alias_table.json`: the table, with a content version `sha256:<12 hex>`.
- `artifacts_trial/alias/alias_build_report.json`: counts and the top merged spellings.
- `artifacts_trial/canonical/<source>/<source>_canonical.jsonl`, `.parquet`, and
  `_canonical_summary.json` for `resumes` and `postings`.
- `artifacts_trial/alias/alias_draft.csv` and `alias_draft_summary.json` (propose only).

Each canonical record keeps every structured field, replaces `skills`, keeps the original list in
`skills_raw`, and carries the table version in `alias_table_version`. Reruns with the same inputs
produce byte-identical files.

## Curation

`alias-table propose` only writes a draft. It never edits `alias_curated.csv`. Review
`alias_draft.csv` line by line, copy the rows you accept (`alias,canonical`) into
`src/getwork/alias/alias_curated.csv`, then rerun `alias-table build` and `alias-table apply`.
A draft row with status `conflict` names an alias that was proposed with two different canonicals.

## Trial and full-corpus tables differ

The table is built from the structured records of the chosen directory, so the trial table and a
full-corpus table are different tables with different versions. Do not mix them: the version is
stamped on every canonical record, and later stages refuse to continue when they see two versions.

## Checklist

### AU-1: build the table

Run `uv sync`, then `uv run alias-table build`. Pass: exit 0, and
`artifacts_trial/alias/alias_build_report.json` lists merged forms. Skim `top_merges` for wrong
merges (case variants and curated rows only).

### AU-2: apply the table

Run `uv run alias-table apply`. Pass: exit 0, the two summaries show `distinct_after` below
`distinct_before` and few `n_records_without_skills`. Open a few canonical records and compare
`skills` with `skills_raw`.

### AU-3: check git status

Run `git status`. Pass: nothing under `artifacts_trial/`, `data/structured/`, or `data/processed/`
appears as changed.

### AU-4: optional LLM proposals (uses your key)

Configure `.env` (Voyager or the local backend, see the structurer runbook), then run
`uv run alias-table propose --model <id>`. Review `alias_draft.csv`, copy the rows you accept into
`src/getwork/alias/alias_curated.csv`, and rerun AU-1 and AU-2. Pass: only reviewed rows are added,
the rerun exits 0, and `alias_table_version` changes.

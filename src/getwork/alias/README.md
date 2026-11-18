# Alias table runbook

The structurer stores skills exactly as the model wrote them, so one skill shows up as `JAVA`,
`Java`, `java`, or `JS` and `JavaScript`. `alias-table` merges those variants into one canonical
spelling per skill and rewrites every structured record through the table. Everything downstream
(counting, clustering, lookup, comparison) reads the canonical records, not the structured ones.

The table has three layers, applied in this order. First, every spelling in the corpus is grouped
by a normalized key (case-folded, whitespace collapsed, trailing punctuation removed) and the group
takes its best spelling. Second, the version families in `src/getwork/alias/version_families.csv`
fold versioned spellings of a reviewed product into one canonical skill (`Windows 7`, `Windows
Server 2019` and `Windows 11` all become `Windows`). Third, the curated file
`src/getwork/alias/alias_curated.csv` (`alias,canonical` rows) overrides both for real
abbreviations (`js` becomes `JavaScript`). Conflicting curated rows or version families fail the
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

`alias-table build` takes `--curated PATH` to use another curated CSV and `--families PATH` to use
another version families CSV (a file with only the header disables folding). `alias-table propose`
takes the same `--backend`, `--model`, `--env-file`, and `--extra-body` options as the structurer,
plus `--batch-size`, `--reference-size`, `--max-skills`, and `--max-tokens`.

Exit codes: 0 ok, 1 propose finished with failed batches, 2 usage or config error (missing input,
missing table, corrupt record, bad curated file), 3 fatal LLM error.

## Output files

- `artifacts_trial/alias/alias_table.json`: the table, with a content version `sha256:<12 hex>`.
- `artifacts_trial/alias/alias_build_report.json`: counts and the top merged spellings.
- `artifacts_trial/alias/alias_version_report.json`: `alias_table_version`, `n_families`,
  `n_version_forms_folded`, `top_version_folds` (canonical and the spellings folded into it) and
  `version_candidates` (`skill`, `base`, `count`: spellings that end in a version-like number whose
  base is a skill but that no family folded, by document frequency).
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

## Version families

`version_families.csv` has the columns `base,canonical,map_bare`. A skill folds to `canonical` when
its normalized key is the normalized `base` followed by a version expression: a number (`7`, `2008`,
`3.9`, `1.x`, `11g`), a release name (`xp`, `vista`, `nt`, `ce`), an optional release suffix (`R2`,
`SP3`), optionally with a `v` or `version` prefix, and lists such as `7/8/10`. Nothing else may
follow: `Windows 7 Professional`, `Java ME`, `Windows Phone` and `SQL Server Express` do not fold.
`map_bare=no` keeps the bare base out of the family (`sql` is SQL, `server` is not a skill), so only
its versioned forms fold. The folds are applied when the table is built, and a curated row wins over
a family fold.

Only reviewed products are listed, because a generic "strip the trailing number" rule is wrong here.
Hardware models (`Cisco 3750`, `Nexus 7K`) and standards (`ISO 27001` against `ISO 9001`) are never
families. Open `alias_version_report.json` after a build and add one row per real version family
from `version_candidates`. A version-type row in an `alias-table propose` draft becomes a family row
too, and a spelling row becomes a curated row.

## After changing the alias table

Changing the families file or the curated file changes `alias_table_version`. Every later stage
checks that version, so each artifact below refuses to be used with the new table and must be
regenerated, in this order. `bash scripts/bash/run_pipeline.sh` runs all of it. Nothing is migrated;
the rerun overwrites existing files. To keep the old results for comparison, set a fresh root such
as `ARTIFACTS_DIR=artifacts_trial/v2` first; the skill-embedding cache is then not reused.

| # | Command | Why it is stale | Real calls |
|---|---------|-----------------|------------|
| 1 | `alias-table build` | new table and version | none |
| 2 | `alias-table apply` | canonical records carry the old version | none |
| 3 | `taxonomy counts` | counts and the top-1,000 selection change | none |
| 4 | `taxonomy embed` | the selected skill list changed | Voyager (user-run) |
| 5 | `taxonomy grid` (then optionally `taxonomy select`) | labels belong to the old vectors | none |
| 6 | `taxonomy name --method llm --force`, then edit `cluster_names.csv` | cluster ids and members changed | LLM (user-run) |
| 7 | `taxonomy build` | new taxonomy version | none |
| 8 | `relabel` | records carry the old alias and taxonomy versions | none by default |
| 9 | `doc-embed build`, `doc-embed embed` | document texts contain cluster names and skills | Voyager (user-run) |
| 10 | `retrieve run` | rankings read the new vectors | none |

Anything built on the old rankings (hand labels, judge outputs, a running web app) must be redone
or restarted; the web app reads the artifact tree it was started with.

## Trial and full-corpus tables differ

The table is built from the structured records of the chosen directory, so the trial table and a
full-corpus table are different tables with different versions. Do not mix them: the version is
stamped on every canonical record, and later stages refuse to continue when they see two versions.

## Checklist

### AV-1: build with version families

Run `uv run alias-table build --structured-dir data/structured --artifacts-dir ROOT`, then open
`ROOT/alias/alias_version_report.json`. Pass: exit 0, and `top_version_folds` shows Windows, SQL
Server, Oracle and others with their variants. Skim `version_candidates`; add a row to
`version_families.csv` for each real version family, never for hardware models or standards, and
rerun until the list is short.

### AV-2: apply and check the counts

Run `uv run alias-table apply ...`, then `uv run taxonomy counts --artifacts-dir ROOT` and list the
skills starting with `windows` from `ROOT/skills/skill_counts.parquet`. Pass: no skill matches
`Windows <digit>` or `Windows Server <digit>`; `Windows Server 2019` and the other server variants
are the single skill `Windows`.

### AV-3: optional LLM proposals with the new prompt (uses your key)

Run `uv run alias-table propose --model <id>`. Review `alias_draft.csv`: a version-type row becomes
a family row, a spelling row becomes a curated row. Pass: only reviewed rows are added and the build
still exits 0.

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

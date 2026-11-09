# relabel (pipeline step 5)

Gives every document skill a cluster name (`Other` for noise and for skills nothing fits) and
writes the relabeled records. A relabeled record's `skills` list is the final skill set used by
document embedding and retrieval; `skill_clusters[i].cluster` is the name used by the Full
serialization.

## Modes

- **lookup_only** (default, free, offline): known skills get their cluster from the taxonomy,
  unseen skills go to `Other` with `via: "default"`, and incomplete documents are flagged in
  `incomplete_reason` (`no_skills`, `few_skills` below the per-source 10th percentile,
  `low_coverage` below `--min-known`) but not reprocessed.
- **llm** (`relabel --use-llm`, user-run): an LLM assigns each distinct unseen skill to a cluster
  or `Other` (`via: "llm"`), and re-reads the text of incomplete documents to add skills.
  The taxonomy lookup always wins over the model. Document embedding refuses to combine sources
  relabeled in different modes.

## Command

```
relabel [--source resumes|postings|both] [--artifacts-dir artifacts_trial]
        [--processed-dir data/processed] [--use-llm] [--min-known 0.5]
        [--max-input-chars 48000] [--unseen-batch-size 50] [--allow-mixed]
        [--backend ... --model ... --env-file .env --extra-body JSON]
```

Defaults point at the trial tree (`artifacts_trial`). The LLM options matter only with
`--use-llm`. Exit codes: 0 ok, 1 finished with failures (unresolved unseen skills or failed
re-extractions), 2 usage, config or input error, 3 fatal LLM error (caches are saved first).

Reads (never writes): `alias/alias_table.json`, `taxonomy/taxonomy.json`,
`canonical/<source>/<source>_canonical.jsonl`, and with `--use-llm` the `text_clean` column of
`<processed-dir>/<source>/<source>_clean.parquet`.

## Outputs

Per source under `relabeled/<source>/`: `<source>_relabeled.jsonl`, `.parquet`,
`_relabeled_summary.json`, `<source>_relabel_failures.csv` (header `id,stage,message`). With
`--use-llm` also `relabeled/_cache/unseen_assignments.json` and `reextract.json`. Rerunning with
the same inputs and a warm cache gives byte-identical files.

Record fields added to the canonical ones (CC-15): `added_skills`, `skill_clusters`
(`skill`, `cluster`, `via` in taxonomy/noise/llm/default), `incomplete_reason`, `reextracted`,
`relabel_mode`, `taxonomy_version`, `relabel_model`, `relabel_backend`,
`relabel_prompt_version`.

## Caveats

- The trial taxonomy is small, so many trial documents are flagged `low_coverage`. That is
  expected.
- Cache entries carry the taxonomy version, model, backend and prompt version. Entries from
  another taxonomy version are dropped. Entries from another model, backend or prompt version
  raise `RelabelProvenanceError` (exit 2). Use a fresh cache directory when you change model, or
  pass `--allow-mixed` (not for a dataset you will report).

## User-run checklist

### RU-1 Lookup-only run

`uv run relabel` (both sources, trial defaults). Needs `taxonomy.json` and the canonical files.
Pass: exit 0; both summaries show `relabel_mode: lookup_only`, `via_counts` and `n_incomplete`.
Open a few records and read `skill_clusters`.

### RU-2 Estimate the LLM work (optional)

Configure the backend (`.env`; Voyager or the local server). The RU-1 summary gives
`n_unseen_distinct` and `n_incomplete`: about `n_unseen_distinct / 50` batch calls plus one call
per incomplete document. For a local server, make sure it is running and the embedding model is
not holding the GPU.

### RU-3 LLM run (optional)

`uv run relabel --use-llm --model <id> --source resumes`, then the same for postings. If it
stops, rerun the same command: finished batches and documents come from the cache.
Pass: exit 0 (or 1 with a short failures CSV), `relabel_mode: llm`, `relabel_model` set,
`n_unseen_unresolved` small. Read ten `added_skills` against the source text.

### RU-4 One model for both sources (optional)

Use one model and backend for both sources. Switching raises `RelabelProvenanceError` unless you
pass `--allow-mixed`. Pass: both summaries name the same model and backend.

### RU-5 Working tree check

`git status`. Nothing under `artifacts_trial/`, `data/structured/`, or `data/processed/` appears
as changed by an unintended run (the relabeled outputs are the only new files).

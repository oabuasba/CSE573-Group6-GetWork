# docembed (pipeline step 6)

Turns each relabeled record into two texts (Full and B2), gives it an experience level, and
embeds the texts into unit-normalized document vectors that two-way retrieval multiplies. The
`build` step needs no model and runs offline; the `embed` step is user-run (GPU).

## Text format

Both systems start with the same two lines. Missing title or level prints `Not specified`, no
skills prints `Skills: none`, and whitespace inside a field collapses to single spaces.

Full (skills grouped under cluster names, groups ordered by taxonomy cluster id, `Other` last):

```
Title: Backend Engineer
Level: Associate
Skills: Front-end: Vue, CSS; Cloud: AWS; Other: Excel
```

B2 (the same text without cluster labels and group separators):

```
Title: Backend Engineer
Level: Associate
Skills: Vue, CSS, AWS, Excel
```

The wording is part of `SERIALIZATION_VERSION` (`ser_v1`); changing it invalidates old vectors.
No retrieval instruction prefix is used: both sides are embedded as plain text.

## Levels

Scale (LinkedIn's): Internship, Entry level, Associate, Mid-Senior level, Director, Executive,
plus Not specified. Postings use the listed level (`level_source: listed`); when it is absent or
unrecognized the level derives from years of experience (`years`), else Not specified
(`missing`). Resumes always derive from years. Default `--level-bands 2,5`: below 2 years Entry
level, 2 up to 5 Associate, 5 or more Mid-Senior level. Internship, Director and Executive are
never derived. This answers Q-3 as a proposal; confirm or change the bands before the
full-corpus run.

## B2 order (Q-10)

`--b2-order grouped` (default) keeps the Full text's grouped skill order, which is literally
"the same text without the labels" (ADR-5) but lets adjacency carry some cluster information to
B2. `--b2-order record` keeps each record's own skill order for a cleaner ablation. The choice is
stored in every summary and vector meta and must match across sources.

## Commands

```
doc-embed build [--source resumes|postings|both] [--artifacts-dir artifacts_trial]
                [--processed-dir data/processed] [--level-bands 2,5]
                [--b2-order grouped|record]
doc-embed embed [--source both] [--artifacts-dir artifacts_trial] [--chunk-size 256]
                [--no-cache] [--embed-backend ... --embed-model ... --device ... --dtype ...
                --embed-batch-size ... --max-seq-length ... --env-file .env]
```

Defaults point at the trial tree (`artifacts_trial`). Exit codes: 0 ok, 2 usage, configuration
or input error (nothing written), 3 embedding error (cached chunks kept).

`build` reads `taxonomy/taxonomy.json`, `relabeled/<source>/<source>_relabeled.jsonl` and, for
postings, the `id` and `experience_level` columns of
`<processed-dir>/postings/postings_clean.parquet` (a missing Parquet is not an error: levels
fall back to years; an unreadable one is an input error). It refuses records whose alias
table or taxonomy version differs from the taxonomy file, and sources relabeled in different
modes, models or backends.

## Outputs

Per source under `documents/<source>/`: `<source>_documents.jsonl`, `.parquet`,
`_documents_summary.json` (from build), and `<source>_full.npy`, `_full_ids.json`,
`_full_meta.json` plus the same three for `_b2` (from embed). Chunk caches live in
`documents/<source>/_embed_cache/` and are shared by both systems (`--no-cache` skips them).
Each meta records the embedding identity, taxonomy, alias table, relabel mode, serialization,
bands, B2 order, and a `texts_hash` so retrieval can refuse stale or mixed vectors.

## Resuming

If `doc-embed embed` stops (out of memory, interrupted), run the same command again: cached
chunks are reused and vectors are written only after both systems of a source are embedded.
If it stops on the second source, the first source's vectors are already written and valid; the
rerun reuses them from the cache and continues.

## User-run checklist

### DU-1 Build the documents (no key, no GPU)

Run `uv run doc-embed build`. Pass: exit 0; the summaries show `level_counts` and
`level_source_counts`, and `examples` shows readable Full and B2 texts. Open
`documents/postings/postings_documents.jsonl` and check that `level` matches the listed level of
a few postings in the clean Parquet, and that resume levels follow the 2/5 bands.

### DU-2 Free the GPU and run the preflight

Stop any local LLM server (the 4B embedding model and the LLM do not fit together on 12 GB), then
run `uv run embed-preflight`. Pass: the preflight passes.

### DU-3 Embed

Run `uv run doc-embed embed`; if it stops, rerun the same command. Pass: exit 0; each meta file
shows model `Qwen/Qwen3-Embedding-4B`, backend `local`, dim and `texts_hash`; the printed summary
shows `n` equal to the document count.

### DU-4 Check the geometry (optional)

In a Python shell load a source's Full vectors with
`getwork.docembed.vectors.read_system_vectors` and confirm the row norms are 1 and a document's
nearest neighbor within the same source is plausible. Pass: norms within 1e-3 of 1.

### DU-5 Check the working tree

Run `git status`. Pass: nothing under `artifacts_trial/`, `data/structured/` or `data/processed/`
appears as changed.

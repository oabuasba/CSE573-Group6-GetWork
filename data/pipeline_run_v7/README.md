# Pipeline run v7

The artifact tree behind the preliminary evaluation and the slides: alias table, canonical
records, skill vectors, taxonomy (k-means, 10 clusters), relabeled records, document records,
and top-k retrieval results for B1+, B2 and Full. Built from `data/structured` (10,567 resumes,
8,416 postings) with `pipeline_run.txt` as the log of the original run (it was produced under
`artifacts_trial/v7`, so the log shows that path).

## Committed

Everything under 50 MB: `alias/`, `canonical/`, `relabeled/`, `taxonomy/`, `retrieval/`,
`skills/` (counts, `skill_vectors.npy`, ids, meta), `documents/**/*.jsonl` and `pipeline_run.txt`.

## Not committed (git-ignored)

| File | Size |
| --- | --- |
| `documents/postings/postings_b2.npy` | 138 MB |
| `documents/postings/postings_full.npy` | 138 MB |
| `documents/resumes/resumes_b2.npy` | 173 MB |
| `documents/resumes/resumes_full.npy` | 173 MB |
| `*/_embed_cache/` (embedding request caches) | reproducible |

Without the four `.npy` files the document vectors are missing, so steps that read them (the web app, `retrieve run`) fail until you rebuild them.

## Rebuild the .npy files

Needs an ASU Voyager key in your environment (real embedding calls; never commit it). This
re-embeds only the committed document records with the same model, in place:

```
uv run doc-embed embed --artifacts-dir data/pipeline_run_v7 --embed-backend voyager --embed-model qwen3-embedding-8b
```

Note: `ARTIFACTS_DIR=<new folder> bash scripts/bash/run_pipeline.sh kmeans-01` reruns the whole
pipeline (k=10, top 1,000 skills) and regenerates everything, including the LLM-written cluster
names, so its `.npy` files must not be mixed with the committed documents here. Use it only to
build a separate run.

# Skill taxonomy runbook

The taxonomy maps every canonical skill to a named cluster (ADR-6). Related skills then earn
partial credit in the Full system. The `taxonomy` command counts canonical skills across both
corpora, keeps the most frequent, embeds each one alone, reduces with UMAP, searches a clustering
grid (DBSCAN, HDBSCAN, k-means), picks the highest mean silhouette under a noise cap, and writes
`taxonomy.json` after a human review of the cluster names.

Every subcommand takes `--artifacts-dir` (default `artifacts_trial`, git-ignored), reads only
what the previous step wrote, prints a JSON summary to stdout, and creates nothing when an input
is missing (exit 2 naming the path). Nothing under `data/structured/` or `data/processed/` is
written.

## Commands

```
uv run taxonomy counts [--top-n 1000] [--min-df 2]
uv run taxonomy embed [embed options] [--chunk-size 256]
uv run taxonomy grid [--max-noise 0.30] [--min-clusters 2] [--seed 0] [--grid-file PATH]
                     [--umap-components 5] [--umap-neighbors 15]
uv run taxonomy select --config-id ID
uv run taxonomy name [--method heuristic|llm] [--top-members 25] [--force]
uv run taxonomy build [--names PATH]
```

- `taxonomy counts` reads `canonical/<source>/<source>_canonical.jsonl` for both sources (all
  records must carry one `alias_table_version`) and writes `skills/skill_counts.parquet` and
  `skills/counts_summary.json`. A skill counts once per document; selection keeps
  `df_total >= --min-df`, then the first `--top-n`.
- `taxonomy embed` embeds the selected skills (exact canonical string, no prefix) through the
  embedding client (see `src/getwork/embed/README.md`) and writes `skills/skill_vectors.npy`,
  `skill_vectors_ids.json`, `skill_vectors_meta.json`. Chunks are cached in
  `skills/_embed_cache/`, so a rerun makes no client calls.
- `taxonomy grid` refuses vectors from another counts run, reduces with UMAP (`--umap-components`
  5, `--umap-neighbors` 15), runs every configuration, and writes `taxonomy/grid_context.json`,
  `grid_results.csv`, `grid_labels.npy`. If a configuration is eligible it also writes
  `selection.json` and `review_<config_id>.csv`; otherwise it exits 2 with the grid files kept.
  `--grid-file` is a JSON object overriding any of `dbscan_eps`, `dbscan_min_samples`,
  `hdbscan_min_cluster_size`, `kmeans_k`. The defaults are proposals (open question Q-4).
- `taxonomy select` rewrites `selection.json` (`source: manual`) and the review file for another
  configuration; a skipped or unknown id is exit 2.
- `taxonomy name` writes `taxonomy/cluster_names.csv` (columns `cluster_id, name, size,
  top_members, method, status, model, backend, prompt_version`). The heuristic joins the three
  most frequent members; `--method llm` asks the model one question per cluster (options
  `--backend`, `--model`, `--env-file`, `--extra-body`). Clusters the model fails get a heuristic
  name, method `heuristic_fallback`, and exit 1. An existing file is never overwritten without
  `--force`, so hand edits are safe.
- `taxonomy build` reads the edited names file, the selection, and the labels, and writes
  `taxonomy/taxonomy.json`.

`name` and `build` both refuse a `selection.json` that does not belong to the current grid files
or to the current counts (for example after a grid run that found no eligible configuration, or
after rerunning `counts` and `embed` without rerunning `grid`): exit 2, nothing written.

Exit codes: 0 ok, 1 `name` used heuristic fallbacks, 2 usage, config or input error, 3 fatal
LLM or embedding error.

## Review procedure

1. Open `review_<config_id>.csv` and `grid_results.csv`; check that clusters are recognizable
   and that `largest_cluster_fraction` is not one giant cluster.
2. Optionally run `taxonomy select --config-id <other id>`.
3. Run `taxonomy name`, then edit the name cells in `cluster_names.csv`: no `:`, `;`, `,`, unique,
   not "Other".
4. Run `taxonomy build`.

Merging or splitting clusters by hand is not supported in this round; pick another configuration
or change the grid.

## Trial caveat

The trial corpus (100 resumes and 100 postings) is tiny, so the trial taxonomy is rough. A
taxonomy built on the trial must not be used with full-corpus runs: the alias table version and
the taxonomy version are stamped into downstream artifacts and will not match.

## User-run checklist

### TU-1: counts

`uv sync`, then `uv run taxonomy counts` (needs the alias-table apply output). Pass: exit 0;
`counts_summary.json` shows `n_selected`, `mention_coverage`, `one_corpus_only_fraction`. On the
trial `n_selected` is a few hundred; that is expected.

### TU-2: embed

Stop any local LLM server, then `uv run taxonomy embed` (the first model use downloads
Qwen3-Embedding-4B; see embedding-client EU-1 to EU-3). Pass: exit 0; `skill_vectors.npy` exists;
`nvidia-smi` showed no out-of-memory crash.

### TU-3: grid

`uv run taxonomy grid` (the first UMAP call compiles with numba and can take about a minute).
Pass: exit 0 and `selection.json`. On "no eligible configuration" raise `--max-noise` or edit the
grid (`--grid-file`). Open `grid_results.csv` and the review file.

### TU-4: LLM names (optional)

Configure `.env` (Voyager or the local backend), then `uv run taxonomy name --method llm --model
<id>`. Pass: `cluster_names.csv` has method `llm` for most rows; read every name.

### TU-5: heuristic names

Without an LLM: `uv run taxonomy name`. Edit the names until each is a clear label. Pass: no row
has status `duplicate`.

### TU-6: build

`uv run taxonomy build`, then `git status`. Pass: exit 0; `taxonomy.json` lists your names;
nothing under `artifacts_trial/`, `data/structured/`, or `data/processed/` shows as changed.

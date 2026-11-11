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
uv run taxonomy name [--method heuristic|llm] [--top-members 25] [--workers 1] [--force]
uv run taxonomy build [--names PATH] [--allow-member-names]
```

- `taxonomy counts` reads `canonical/<source>/<source>_canonical.jsonl` for both sources (all
  records must carry one `alias_table_version`) and writes `skills/skill_counts.parquet` and
  `skills/counts_summary.json`. A skill counts once per document; selection keeps
  `df_total >= --min-df`, then the first `--top-n`.
- `taxonomy embed` embeds the selected skills (exact canonical string, no prefix) through the
  embedding client (see `src/getwork/embed/README.md`) and writes `skills/skill_vectors.npy`,
  `skill_vectors_ids.json`, `skill_vectors_meta.json`. Chunks are cached in
  `skills/_embed_cache/`, so a rerun embeds no cached chunk (Voyager still sends one probe request at start-up).
- `taxonomy grid` refuses vectors from another counts run, reduces with UMAP (`--umap-components`
  5, `--umap-neighbors` 15), runs every configuration, and writes `taxonomy/grid_context.json`,
  `grid_results.csv`, `grid_labels.npy`. If a configuration is eligible it also writes
  `selection.json` and `review_<config_id>.csv`; otherwise it exits 2 with the grid files kept.
  `--grid-file` is a JSON object overriding any of `dbscan_eps`, `dbscan_min_samples`,
  `hdbscan_min_cluster_size`, `kmeans_k`. The defaults are proposals (open question Q-4).
- `taxonomy select` rewrites `selection.json` (`source: manual`) and the review file for another
  configuration; a skipped or unknown id is exit 2.
- `taxonomy name` writes `taxonomy/cluster_names.csv` (columns `cluster_id, name, size,
  top_members, method, status, model, backend, prompt_version`). `--method llm` asks the model
  one question per cluster, in id order (options `--backend`, `--model`, `--env-file`,
  `--extra-body`), with the prompt `name_v2`. Clusters the model fails get a heuristic name,
  method `heuristic_fallback`, and exit 1. The heuristic joins the three most frequent members;
  that is a list of member skills, so heuristic names are for offline runs only. An existing file
  is never overwritten without `--force`, so hand edits are safe.
- `taxonomy build` reads the edited names file, the selection, and the labels, and writes
  `taxonomy/taxonomy.json`. It exits 2 naming the cluster ids (never the text) and writes nothing
  when any name is a list of member skills, unless `--allow-member-names` is given.

## Concept-label names

The cluster name goes into every document's embedded text, so a name that just lists members
repeats skills and carries no concept. `name_v2` asks for a short concept label (one to four
words, for example "Unit Testing" for JUnit, Selenium and Mockito) that differs from the names
already given. The model's reply is rejected, with feedback and a retry, when the name is generic
(Tools, Other, Cluster 4), longer than 5 words, a list of member skills, equal to one member
skill, or equal (ignoring case) to a name already used. A name is a member list when it splits on
`/`, `|`, `&`, `+` or "and" (or on spaces) into two or more parts and more than half of the parts
are member skills. In `cluster_names.csv` the `status` column is `member_list` for such a name,
else `duplicate` for a repeated name, else `ok`; `taxonomy name` prints `n_member_list`. A
human-edited single-member name (for example Python) is accepted by `taxonomy build`.

`name` and `build` both refuse a `selection.json` that does not belong to the current grid files
or to the current counts (for example after a grid run that found no eligible configuration, or
after rerunning `counts` and `embed` without rerunning `grid`): exit 2, nothing written.

Exit codes: 0 ok, 1 `name` used heuristic fallbacks, 2 usage, config or input error, 3 fatal
LLM or embedding error.

## Parallel naming

`taxonomy name --method llm --workers N` (default 1) asks the model about N clusters at once.
With `--workers 1` clusters are asked in id order and every request lists the names already used.
With 2 or more, each cluster is first asked without the used-names list (member-list replies are
still retried with feedback); the answers are then checked in id order, and a name that repeats an
earlier cluster's is re-asked once, in id order, with the used-names list. The lower id keeps a
contested name, so the names are the same for every `--workers` of 2 or more, whatever the finish
order, but can differ from `--workers 1`. One rate-limit (429) pause is shared by all workers.

Each cluster prints one progress line on stderr when its name is final, in id order; stdout stays
the single JSON summary line (`2>/dev/null` silences the progress):

```
named 7/71: cluster 12 -> "Unit Testing" (attempts 2)
fallback 8/71: cluster 13 (replies rejected, attempts 3)
```

`--workers` is accepted and ignored for `--method heuristic`. `scripts/run_pipeline.sh` passes
`NAME_WORKERS` (default 4).

## Review procedure

1. Open `review_<config_id>.csv` and `grid_results.csv`; check that clusters are recognizable
   and that `largest_cluster_fraction` is not one giant cluster.
2. Optionally run `taxonomy select --config-id <other id>`.
3. Run `taxonomy name --method llm`, then edit the name cells in `cluster_names.csv`: each name is
   a concept label, not a list of member skills; no `:`, `;`, `,`, unique, not "Other".
4. Run `taxonomy build`.

Merging or splitting clusters by hand is not supported in this round; pick another configuration
or change the grid.

## Plots

`uv run taxonomy plot` draws the chosen clustering and how the skills relate to it. It reads
saved artifacts only (no API calls, nothing is embedded) and writes to
`<artifacts-dir>/taxonomy/plots/` (change with `--out-dir`). Open `index.html`: it embeds every
plot. Options: `--artifacts-dir`, `--out-dir`, `--seed` (seed of the 2-D map, default 0),
`--no-interactive` (skip the 5 MB standalone `cluster_map.html`), `--max-label-chars`
(cluster-name length on the map, default 22). Exit 0 on success; exit 2 with a message naming the
missing file and the command to run (`taxonomy embed`, `grid`, `name`) when an input is missing or
does not belong together (stale selection, vectors of another skill list, names of another
clustering). On an input error (exit 2 before drawing) nothing is written; an I/O error while writing
(for example a full disk) can leave a partial set of files, each of them complete. Run it after `taxonomy name` (it reads
`cluster_names.csv`, so edited names show up without `taxonomy build`).

How to read each output:

- `cluster_map.png` / `cluster_map.html`: every skill as a dot, colored by cluster, noise in grey,
  cluster names near the centroids (leader lines when labels had to move apart). Positions come
  from a separate 2-component cosine UMAP of the same vectors, so distances are not those of the
  5-D clustering space: a cluster may look split and neighbours may be far apart. In the HTML,
  hover shows skill, cluster, `df_resumes` and `df_postings`; click a legend entry to hide a
  cluster.
- `cluster_sizes.png`: skills per cluster, largest first, noise bar on top. A long tail of tiny
  clusters or a dominant bar means the configuration is lopsided.
- `cluster_corpus_balance.png`: one point per cluster (size = skills), x = summed resume
  mentions, y = summed posting mentions, log axes (a count of 0 sits on the lowest line,
  labelled 0). The dashed line is the overall postings-to-resumes ratio; points far below it are
  resume-heavy, points on the lowest line are never mentioned in postings.
- `cluster_one_corpus.png`: per cluster, how many skills occur in both corpora, in resumes only
  or in postings only (title shows the overall one-corpus share). Clusters with long colored
  segments only describe one corpus.
- `skill_document_frequency.png`: each skill at (`df_resumes`, `df_postings`), colored by
  cluster, noise grey, top skills named. Dots on the lowest line (0) occur in one corpus only.
- `grid_overview.png`: every grid configuration as clusters vs noise fraction, colored by
  silhouette (marker per method); the chosen configuration is ringed in red. Skipped
  configurations are left out and counted in the title.
- `cluster_summary.csv`: the numbers behind the cluster plots (size, summed df per corpus, skill
  counts by corpus).

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

### TU-5: heuristic names (offline runs only)

Without an LLM: `uv run taxonomy name`. Heuristic names normally get status `member_list`: rewrite
each into a concept label, or pass `--allow-member-names` to `taxonomy build`. Pass: no row has
status `duplicate`.

### TU-6: build

`uv run taxonomy build`, then `git status`. Pass: exit 0; `taxonomy.json` lists your names;
nothing under `artifacts_trial/`, `data/structured/`, or `data/processed/` shows as changed.

### AV-4: full pipeline with LLM names (uses your keys)

`bash scripts/run_pipeline.sh` makes embedding and LLM calls; `NAME_METHOD` defaults to `llm`
(set the model in `.env` or run `taxonomy name` by hand with its options). `NAME_METHOD=heuristic`
is for offline runs only and adds `--allow-member-names` to `taxonomy build`. Pass: every step
exits 0 (or 1 from `name` if some clusters fell back, see AV-5).

### AV-5: read the names

Open `ROOT/taxonomy/cluster_names.csv`. Pass: every row has method `llm` and status `ok`, and each
name is a concept label. A `heuristic_fallback` row (status `member_list`) must be edited by hand
before `taxonomy build` succeeds.

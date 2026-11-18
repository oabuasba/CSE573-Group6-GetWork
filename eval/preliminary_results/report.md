# Evaluation report

## Run

- Seed: 0
- Pool: software and adjacent IT postings, capped at 5000 by a seeded sample
- Resumes: 10567; postings: 8416
- Taxonomy version: sha256:02456e823462
- Relabel mode: lookup_only
- B2 text order (b2_order): grouped
- B1+ weights (dev): skills 0.700, title 0.150, level 0.150; source default
- B1+ weights (test): skills 0.700, title 0.150, level 0.150; source default
- Parameters: dev_postings 10, dev_resumes 10, human_postings 10, human_resumes 10, seed 0, test_postings 30, test_resumes 30

## Judges

- Rubric hash: sha256:87ecca1b14a7
- Judge kappa, dev pairs: 0.480
- Judge kappa, test pairs: 0.547

| Judge | Model | Family | Truncated pairs |
|---|---|---|---|
| judge-a | gemma4-31b-it | google | 19 |
| judge-b | llama4-maverick-17b | meta | 19 |

## Ranking quality (judge labels, test queries)

### resume_to_posting (30 queries)

| System | P@5 | P@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| B1+ | 0.367 [0.267, 0.467] | 0.340 [0.257, 0.423] | 0.594 [0.450, 0.735] | 0.703 [0.646, 0.753] |
| B2 | 0.527 [0.413, 0.640] | 0.507 [0.407, 0.603] | 0.635 [0.505, 0.760] | 0.820 [0.765, 0.865] |
| Full | 0.460 [0.347, 0.573] | 0.447 [0.340, 0.553] | 0.632 [0.486, 0.767] | 0.783 [0.726, 0.832] |

### posting_to_resume (30 queries)

| System | P@5 | P@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| B1+ | 0.227 [0.133, 0.333] | 0.167 [0.090, 0.260] | 0.384 [0.246, 0.529] | 0.639 [0.548, 0.726] |
| B2 | 0.387 [0.253, 0.527] | 0.313 [0.193, 0.443] | 0.530 [0.369, 0.692] | 0.834 [0.772, 0.888] |
| Full | 0.327 [0.213, 0.453] | 0.290 [0.187, 0.407] | 0.478 [0.331, 0.631] | 0.798 [0.734, 0.857] |

## System gaps

### resume_to_posting

| Gap | Metric | Diff | 95% CI | p |
|---|---|---|---|---|
| B2-B1+ | P@5 | 0.160 | [0.073, 0.253] | 0.002 |
| B2-B1+ | P@10 | 0.167 | [0.097, 0.237] | <0.001 |
| B2-B1+ | MRR | 0.041 | [-0.084, 0.166] | 0.534 |
| B2-B1+ | nDCG@10 | 0.117 | [0.058, 0.172] | <0.001 |
| Full-B2 | P@5 | -0.067 | [-0.140, 0.013] | 0.147 |
| Full-B2 | P@10 | -0.060 | [-0.123, 0.007] | 0.105 |
| Full-B2 | MRR | -0.004 | [-0.104, 0.099] | 0.939 |
| Full-B2 | nDCG@10 | -0.037 | [-0.078, 0.004] | 0.097 |
| Full-B1+ | P@5 | 0.093 | [0.007, 0.187] | 0.084 |
| Full-B1+ | P@10 | 0.107 | [0.030, 0.183] | 0.016 |
| Full-B1+ | MRR | 0.037 | [-0.089, 0.173] | 0.598 |
| Full-B1+ | nDCG@10 | 0.080 | [0.035, 0.129] | 0.001 |

### posting_to_resume

| Gap | Metric | Diff | 95% CI | p |
|---|---|---|---|---|
| B2-B1+ | P@5 | 0.160 | [0.060, 0.267] | 0.008 |
| B2-B1+ | P@10 | 0.147 | [0.067, 0.240] | <0.001 |
| B2-B1+ | MRR | 0.146 | [-0.022, 0.317] | 0.106 |
| B2-B1+ | nDCG@10 | 0.195 | [0.110, 0.284] | <0.001 |
| Full-B2 | P@5 | -0.060 | [-0.133, 0.007] | 0.157 |
| Full-B2 | P@10 | -0.023 | [-0.083, 0.033] | 0.525 |
| Full-B2 | MRR | -0.051 | [-0.139, 0.022] | 0.236 |
| Full-B2 | nDCG@10 | -0.036 | [-0.085, 0.009] | 0.165 |
| Full-B1+ | P@5 | 0.100 | [0.020, 0.187] | 0.035 |
| Full-B1+ | P@10 | 0.123 | [0.050, 0.207] | 0.003 |
| Full-B1+ | MRR | 0.095 | [-0.062, 0.256] | 0.262 |
| Full-B1+ | nDCG@10 | 0.159 | [0.079, 0.245] | <0.001 |

## Human check

_not run yet_

## Taxonomy validity (ARI)

- k: 10; seeds: 5; labeled resumes: 10300 (267 unlabeled excluded)
- Taxonomy: kmeans-01, 10 clusters, silhouette 0.080, noise fraction 0.000
- Vocabulary in one corpus only: 88 of 1000 (0.088)

| Profile | Dims | ARI mean | ARI sd |
|---|---|---|---|
| Cluster profiles | 10 | 0.187 | 0.001 |
| Raw skills | 1000 | 0.203 | 0.002 |

## Caveats

- A pair is relevant (label 2) only when the lower of the two labels, resume to posting and posting to resume, is 2: the lower-label rule is deliberately strict.
- With three systems Kendall's tau between system orders can only be 1, 1/3, -1/3 or -1, so the gap sizes are reported beside it.
- There is no multiplicity correction: three gaps, four metrics and two directions are tested separately and the p-values are not adjusted.
- The relevance labels come from LLM judges, not from people; their agreement with humans is checked on a small set only.
- The human check set is small (tens of pairs), so its kappa and tau values are rough.
- The posting pool is software and adjacent IT postings, a seeded sample of the tech postings table, not all postings.
- B1+ untuned: the test stage used the default B1+ weights, not weights tuned on the dev queries, so the baseline may be understated and the gaps to B2 and Full overstated.
- Human check not run: no author labels or human agreement were used, so the judge labels are unvalidated and these results are preliminary.
- B2 text keeps the Full grouping order (b2_order = grouped, architecture Q-10), so part of the taxonomy effect can appear in B2; rebuild documents with --b2-order record for a cleaner ablation.

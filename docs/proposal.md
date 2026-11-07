::: teaserfigure
*AI Disclosure: Generative AI Tools (Claude) were used in the
preparation of this proposal, including writing and formatting of this
document, researching tools, and verifying data sources and references.*
:::

# Problem Definition {#intro}

Searching for a job and filling a position are two sides of the same
matching problem. Candidates must sift through large volumes of postings
to find the ones they qualify for, while recruiters must screen large
applicant pools to find the candidates who best meet their requirements.
Both tasks are difficult because resumes and job postings often describe
the same skills using different vocabulary. A posting that requires
React experience and a resume that lists single-page applications built
in Vue describe closely related work, yet share few keywords.

Existing approaches address this problem in two main ways. Keyword-based
screening, common in applicant tracking systems, scores candidates by
their exact term overlap with the posting. This approach is transparent,
but gives no credit to related skills and can screen out qualified
applicants whose resumes do not match the exact terms of the posting
[@fuller2021hidden]. Conversely, embedding-based approaches represent
each document as a vector and capture semantic similarity beyond exact
terms [@lavi2021consultantbert]. However, a single document vector makes
it difficult to explain why a match was made, and most job recommender
systems are built for only one direction of the search, recommending
either jobs to candidates or candidates to recruiters, although the
reciprocal nature of the problem has been identified as important
[@deruijt2021review].

To address these limitations, we propose GetWork, a two-way matching
framework for software engineering and adjacent fields. Given a resume,
the system returns the top-$k$ most relevant job postings; given a job
posting, it returns the top-$k$ most relevant candidates. Rather than
relying on exact keywords alone, we derive a skill taxonomy directly
from the corpus by clustering skill embeddings, so that related skills
(e.g., React and Vue) are brought closer together in the matching space,
giving them partial credit. A conversational agent then explains each
match in terms of shared and related skills.

With this framework, we aim to answer the following question: does
structuring resumes and job postings around a corpus-generated skill
taxonomy improve two-way matching quality over both keyword-based
matching and embedding-based matching without the taxonomy, while
producing explanations that candidates and recruiters can act on?

# Data Sets {#data}

We use two publicly available datasets: a corpus of IT resumes for the
candidate side and a collection of LinkedIn job postings for the
employer side \[Table [1](#tab:datasets){reference-type="ref"
reference="tab:datasets"}\].

The resume corpus, released by @jiechieu2021skills, contains 29,783
anonymized resumes collected from online job sites. Each resume is
stored as free text and labeled with one or more of 10 IT occupations:
front-end developer, web developer, software developer, Python
developer, Java developer, database administrator, systems
administrator, network administrator, security analyst, and project
manager. This corpus is well suited to our focus for two reasons. First,
every resume already falls within software engineering or an adjacent
field, so no domain filtering is required on the resume side. Second,
the occupation labels provide a reference point for evaluating whether
our corpus-generated skill clusters align with recognized roles.

The job postings come from the LinkedIn Job Postings dataset
[@arshkon2024linkedin], which contains approximately 124,000 postings
collected across 2023 and 2024. Each posting includes a title, full
description, location, work type, salary range when listed, and
experience level, alongside supplementary tables that link each posting
to its company, industries, and job functions. This dataset complements
the resume corpus in two ways. First, the descriptions are written by
employers, so they reflect the vocabulary that candidates must match.
Second, the title and industry fields allow a rule-based filter to
isolate software engineering postings without an LLM, which keeps the
cost of later structuring steps proportional to the filtered subset
rather than the full dataset. However, the job functions attached to
each posting are broad categories (e.g., Information Technology,
Engineering) rather than specific technologies, so specific skills must
be extracted from the description text.

It is also worth noting the one limitation of this pairing. The resumes
were collected at least three years before the postings, so newer tools
and frameworks may appear frequently in postings but rarely in resumes.
We treat this gap as a realistic source of vocabulary mismatch and
quantify it by reporting the fraction of frequent skills that appear in
only one corpus. Further analysis is needed to determine the size of
each software engineering subset after filtering.

::: {#tab:datasets}
           Resumes             Job postings
  -------- ------------------- --------------------------------
  Source   Jiechieu & Tsopze   LinkedIn (Kaggle)
  Size     29,783              $\sim$`<!-- -->`{=html}124,000
  Period   pre-2021            2023--2024
  Labels   10 IT occupations   Industries, functions
  Text     Full resume         Title, description

  : Summary of the datasets used in this work.
:::

# Methods and Algorithms {#methods}

Our framework is organized into two phases: a build phase that runs once
on the corpus to produce a shared, skill-aware representation of resumes
and job postings, and a query phase that uses this representation to
retrieve and explain matches in both directions. This section introduces
the methods and tools used in each phase
\[Table [2](#tab:tools){reference-type="ref" reference="tab:tools"}\],
while Section [4](#research){reference-type="ref" reference="research"}
describes how they are combined step by step.

::: {#tab:tools}
  Stage                  Tools
  ---------------------- -------------------------------
  Data processing        pandas
  Data structuring       Qwen3-8B, vLLM, Pydantic
  Embeddings             Qwen3-Embedding-4B
  Clustering             scikit-learn, UMAP, (H)DBSCAN
  Retrieval              NumPy
  Conversational agent   LangGraph, CreateAI
  Application            FastAPI, React

  : Tools used in each stage of the framework.
:::

## Data Processing {#methods:data}

Both datasets are loaded and filtered with pandas using rule-based
criteria. Quality filtering is applied to both datasets, removing empty
or very short documents, exact duplicates, and documents with
non-English or corrupted text. Domain filtering is applied only to job
postings, which are retained based on their titles and industry fields.
No large language model (LLM) is used at this stage, which reduces the
corpus before any costly processing is applied.

## Data Structuring {#methods:structuring}

We use LLMs to convert unstructured text into structured records in two
passes. The first pass extracts generic sections (skills, experience,
education, and title) from every filtered document. The second pass
assigns each document's skills to the clusters of our skill taxonomy
(Section [3.4](#methods:cluster){reference-type="ref"
reference="methods:cluster"}). Skills already in the taxonomy are
assigned by direct lookup, while only unseen skills and documents with
incomplete extraction are passed to the LLM with few-shot examples.
Because the first pass covers every filtered document, structuring will
require tens of thousands of LLM calls. Rather than relying on a
rate-limited external API, we host Qwen3-8B [@yang2025qwen3] on GPU
resources provided through ASU's Sol supercomputer, with Google Colab as
a fallback, and serve it with vLLM [@kwon2023vllm], which batches
concurrent requests to increase throughput. This keeps the cost of
structuring fixed and makes results reproducible. Qwen3's reasoning mode
is disabled for both passes, since extraction does not require
multi-step reasoning and the additional output tokens would slow
processing. Each output is validated against a Pydantic schema in JSON
mode and regenerated on failure, which ensures that every document
follows the same structure.

## Embeddings {#methods:embed}

Skills and documents are embedded with Qwen3-Embedding-4B
[@zhang2025qwen3embedding] through the sentence-transformers library
[@reimers2019sbert]. Skills are embedded individually to build the
taxonomy, while each document is embedded as a single vector. Before
embedding, each document is serialized as its title, experience level,
and skills grouped under their cluster names (e.g., "Front-end: Vue,
CSS; Cloud: AWS"), so that related skills share a common label while the
specific skills are preserved. For postings, the experience level is
taken from the listed field; for resumes, it is derived from the total
years of experience extracted in the first pass. For the embedding
baseline, documents use the same skills and format, but skills are
listed without cluster labels. The same model and format are used for
resumes and job postings, so that both share one embedding space.

## Clustering {#methods:cluster}

The skill taxonomy is built by clustering skill embeddings with
scikit-learn [@pedregosa2011sklearn]. DBSCAN [@ester1996dbscan] groups
skills by density: using the cosine distance $d(x, y) = 1 - \cos(x, y)$,
the $\varepsilon$-neighborhood of a skill vector $x$ is
$$\begin{equation}
\label{eq:dbscan}
  N_\varepsilon(x) = \{\, y : d(x, y) \leq \varepsilon \,\},
\end{equation}$$ and $x$ is a core point if
$|N_\varepsilon(x)| \geq \mathit{minPts}$. Clusters are formed by
connecting core points that fall within each other's neighborhoods,
while skills in low-density regions are labeled as noise. Noise skills
receive no cluster label and are kept under an "Other" group in the
document text, so they can still contribute exact matches but receive no
partial credit. Because density estimates degrade in high-dimensional
spaces, we will compare DBSCAN and HDBSCAN [@campello2013hdbscan]
applied after UMAP dimensionality reduction [@mcinnes2018umap], with
k-means as a reference. Cluster quality is measured with the silhouette
score [@rousseeuw1987silhouette], $$\begin{equation}
\label{eq:silhouette}
  s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}},
\end{equation}$$ where $a(i)$ is the mean distance from skill $i$ to the
other skills in its cluster and $b(i)$ is the mean distance to the
skills in the nearest other cluster. Values near 1 indicate
well-separated clusters. Since the silhouette score is undefined for
noise points, it is computed over clustered skills only, in the original
embedding space with cosine distance, and the fraction of skills labeled
as noise is reported alongside it.

## Retrieval {#methods:retrieval}

Matching is based on the cosine similarity between a resume vector
$\mathbf{r}$ and a job vector $\mathbf{j}$: $$\begin{equation}
\label{eq:cosine}
  \mathrm{sim}(\mathbf{r}, \mathbf{j}) = \frac{\mathbf{r} \cdot \mathbf{j}}{\lVert \mathbf{r} \rVert \, \lVert \mathbf{j} \rVert}.
\end{equation}$$ Normalizing every vector to unit length reduces this to
an inner product, so all pairs can be scored at once with NumPy as a
single matrix product, $$\begin{equation}
\label{eq:simmatrix}
  S = \hat{R} \hat{J}^{\top}, \qquad S \in \mathbb{R}^{m \times n},
\end{equation}$$ where $\hat{R}$ and $\hat{J}$ contain the normalized
vectors of $m$ resumes and $n$ job postings. The top-$k$ jobs for
candidate $i$ are the largest entries of row $S_{i,:}$, and the top-$k$
candidates for job $\ell$ are the largest entries of column
$S_{:,\ell}$. As a classical baseline, we score each pair by the Jaccard
overlap of their final skill sets $A_r$ and $A_j$ (after step 5),
ignoring cluster labels: $$\begin{equation}
\label{eq:jaccard}
  J(A_r, A_j) = \frac{|A_r \cap A_j|}{|A_r \cup A_j|} .
\end{equation}$$

## Conversational Agent {#methods:agent}

Explanations are generated by an agent built with LangGraph
[@langgraph2024]. The agent receives a matched resume and posting,
explains the match in terms of shared and related skills, and answers
follow-up questions (e.g., why one job ranked above another) by calling
pipeline functions, such as similarity search, cluster lookup, and
side-by-side comparison, as tools. Unlike the structuring stage, the
agent is interactive and low in volume, so its underlying model is
accessed through the ASU CreateAI platform rather than hosted by us.
CreateAI provides an OpenAI-compatible API, which allows the agent to
use standard LangGraph integrations and to switch models through
configuration alone.

## Application {#methods:app}

The pipeline is implemented in Python and served through a FastAPI
backend. A React frontend allows users to submit a resume or job
posting, view the top-$k$ matches, converse with the agent, and label
spot-check pairs without seeing the judges' labels. Newly submitted
documents are structured through an LLM call, then re-labeled and
embedded (steps 2, 5, and 6) before matching.

# Research Plan {#research}

Our research plan combines the methods in
Section [3](#methods){reference-type="ref" reference="methods"} into an
eight-step pipeline. Steps 1--6 form the build phase and run once on the
corpus, while steps 7--8 form the query phase and run for each request;
newly submitted documents first pass through steps 2, 5, and 6
(Section [3.7](#methods:app){reference-type="ref"
reference="methods:app"}). Each step is labeled by the method it uses
rather than the data it operates on: classical steps rely on rule-based,
statistical, or non-neural machine learning methods, while modern steps
rely on LLMs or neural embeddings. In this division, modern components
produce the representations, while classical components group, rank, and
serve as the baseline.

## Step 1: Filtering (Classical) {#step-1-filtering-classical .unnumbered}

Empty or very short documents, exact duplicates, and documents with
non-English or corrupted text are removed from both datasets. All 10
occupation labels of the resume corpus are retained, as each falls
within software engineering or an adjacent field. Postings are retained
when their title matches software-related terms (e.g., developer,
software engineer), with industry fields used as a secondary check. This
reduces the corpus before any LLM is applied.

## Step 2: Structuring (Classical and Modern) {#step-2-structuring-classical-and-modern .unnumbered}

Qwen3-8B returns each document in a fixed schema containing a title, a
list of skills, and lists of experience entries (with their durations)
and education entries. The extracted skills are then normalized with an
alias table that maps variants to a canonical form (e.g., JS to
JavaScript, k8s to Kubernetes).

## Step 3: Skill Selection and Embedding (Classical and Modern) {#step-3-skill-selection-and-embedding-classical-and-modern .unnumbered}

Each canonical skill is counted across both datasets, and the 1,000 most
frequent skills are retained. Each retained skill is embedded
individually, producing a skill space that is used only for clustering.

## Step 4: Skill Clustering (Classical) {#step-4-skill-clustering-classical .unnumbered}

The skill embeddings are reduced with UMAP, and the parameters of each
clustering method are searched, including $\varepsilon$ and
$\mathit{minPts}$ for DBSCAN
\[Eq. [\[eq:dbscan\]](#eq:dbscan){reference-type="eqref"
reference="eq:dbscan"}\], the minimum cluster size for HDBSCAN, and the
number of clusters for k-means. Among configurations that label fewer
than a maximum fraction of skills as noise, the one with the highest
mean silhouette score
\[Eq. [\[eq:silhouette\]](#eq:silhouette){reference-type="eqref"
reference="eq:silhouette"}\] is selected, and its clusters are reviewed
manually. This review is needed because a high silhouette score
indicates that clusters are well separated, but not that they are
meaningful to candidates or recruiters.

## Step 5: Cluster Re-labeling (Classical and Modern) {#step-5-cluster-re-labeling-classical-and-modern .unnumbered}

Skills already in the taxonomy are assigned to their cluster by direct
lookup, which requires no LLM call; skills labeled as noise in step 4
are assigned to the "Other" group. Skills not found in the taxonomy are
passed to the LLM along with a few example assignments, and are assigned
to "Other" if no cluster fits. Documents for which extraction appears
incomplete, such as those with fewer skills than the 10th percentile of
the corpus or with fewer than half of their skills found in the
taxonomy, are re-processed from their raw text. This fallback captures
skills that are implied rather than listed (e.g., "built CI pipelines"
implies DevOps experience).

## Step 6: Document Embedding (Modern) {#step-6-document-embedding-modern .unnumbered}

Each re-labeled document is serialized in the format described in
Section [3.3](#methods:embed){reference-type="ref"
reference="methods:embed"} and embedded as a single vector, producing a
document space that is separate from the skill space of step 3. The two
spaces are connected only through step 5. Because related skills share a
cluster label in the document text, a resume listing Vue and a posting
requiring React move closer together, while documents that share the
exact skill are expected to remain closer still.

## Step 7: Two-Way Matching (Classical) {#step-7-two-way-matching-classical .unnumbered}

Matches are ranked with the similarity matrix
\[Eqs. [\[eq:cosine\]](#eq:cosine){reference-type="eqref"
reference="eq:cosine"}--[\[eq:simmatrix\]](#eq:simmatrix){reference-type="eqref"
reference="eq:simmatrix"}\], returning the top-$k$ jobs for each
candidate and the top-$k$ candidates for each job. The same retrieval is
applied to documents with the same skills but without cluster labels
(Section [3.3](#methods:embed){reference-type="ref"
reference="methods:embed"}) to produce an embedding baseline. In
parallel, the Jaccard overlap
\[Eq. [\[eq:jaccard\]](#eq:jaccard){reference-type="eqref"
reference="eq:jaccard"}\] between the final skill sets produces a
classical baseline ranking. All three rankings are evaluated on the same
labeled set (Section [5](#evaluation){reference-type="ref"
reference="evaluation"}), so that the difference between the classical
and embedding baselines measures the contribution of the embeddings and
of title and experience level, and the difference between the embedding
baseline and the full system measures the contribution of the taxonomy.

## Step 8: Match Explanation (Modern) {#step-8-match-explanation-modern .unnumbered}

The LangGraph agent is given tools for similarity search, cluster
lookup, and side-by-side comparison, allowing it to explain a match and
answer follow-up questions such as why one job ranked above another.

# Evaluation Plan {#evaluation}

We evaluate (1) the ranking quality of two-way matching, (2) whether the
skill taxonomy is meaningful and gives related skills partial credit,
and (3) the faithfulness of generated explanations. Evaluation combines
a test set labeled by two LLM judges and checked by the authors, and
controlled tests that require no labels.

## Test Set Construction {#eval:testset}

Matching is run against the software engineering postings, capped at a
fixed random sample of 5,000 if the subset is larger. We sample 110
resumes stratified by occupation label
(Section [2](#data){reference-type="ref" reference="data"}), using each
resume's first label when several are listed, so that sampling is
independent of the system being evaluated. Ten resumes form a
development set used to refine the labeling rubric and judge prompts,
and the remaining 100 form a test set used only for final results. For
each test resume, the top-10 postings from every compared system are
pooled, shuffled, and shown without their source system, following the
pooling approach used in TREC [@voorhees2005trec]. Each resume--posting
pair is labeled 0 (not relevant), 1 (partially relevant), or 2
(relevant) based on skills and seniority, and a pair with a clear
seniority mismatch in either direction is labeled 1 at most. Labels are
assigned by LLM judges, as described in
Section [5.2](#eval:judge){reference-type="ref" reference="eval:judge"}.

## LLM-Judged Labels {#eval:judge}

All pooled pairs are labeled by LLM judges using the rubric in
Section [5.1](#eval:testset){reference-type="ref"
reference="eval:testset"}, an approach that has been shown to agree
closely with human relevance labels [@thomas2024llmrel]. Two judges from
different model families label every pair independently, and neither is
a model used in the pipeline or by the agent, as LLM evaluators tend to
favor text they generated themselves [@panickssery2024self]. Judges see
the raw text of each resume and posting, rather than the structured
records, so that extraction errors do not carry into the labels. Each
judge returns a label and a one-sentence rationale. Agreement between
the judges is measured with Cohen's $\kappa$ [@cohen1960kappa]; if it
falls below 0.6, the threshold for substantial agreement
[@landis1977kappa], the rubric is revised on the development set before
the test set is labeled. When the judges disagree, the lower label is
used. As a check against human judgment, both authors assign consensus
labels to 20 pairs (4 resumes with 5 postings each, balanced across
judge labels) without seeing the judges' labels, and we report
percentage agreement and $\kappa$ against the final judge labels; if
$\kappa$ falls below 0.6, the rubric is revised and the test set is
relabeled. The reverse direction (posting to candidates) is evaluated on
20 postings, each ranked against all filtered resumes, using the same
pooling and labeling procedure.

## Metrics and Comparisons {#eval:metrics}

Ranking quality is measured with Precision@K, mean reciprocal rank
(MRR), and normalized discounted cumulative gain (nDCG)
[@jarvelin2002ndcg] \[Table [3](#tab:metrics){reference-type="ref"
reference="tab:metrics"}\]. We compare three configurations. The
classical baseline (B1) ranks postings by the Jaccard overlap between
final skill sets. The embedding baseline (B2) applies cosine matching to
documents with the same skills and format but without cluster labels,
while the full system (Full) applies cosine matching to the
cluster-aware documents. Since all three configurations share the same
skill sets, the difference between B2 and Full isolates the contribution
of the taxonomy.

::: {#tab:metrics}
  Metric                     Measures
  -------------------------- ---------------------------------------------------------------------
  Precision@K ($K{=}5,10$)   Fraction of the top-$K$ results that are relevant
  MRR                        Reciprocal rank of the first relevant result
  nDCG@10                    Graded relevance, discounted by rank
  ARI                        Agreement between cluster-based resume groups and occupation labels
  Explanation rating         LLM-judged rating (1--5) of faithfulness and usefulness

  : Evaluation metrics. For Precision@K and MRR, only a label of 2 is
  treated as relevant.
:::

## Taxonomy Validity {#eval:taxonomy}

The silhouette score [@rousseeuw1987silhouette] measures how compact and
well separated the clusters are, but not whether they are meaningful. We
therefore represent each resume by the number of its skills that fall
into each cluster, group these profiles into 10 groups with k-means, and
report the adjusted Rand index (ARI) [@hubert1985ari] between these
groups and the occupation labels, using each resume's first label as in
Section [5.1](#eval:testset){reference-type="ref"
reference="eval:testset"}. An ARI of 0 indicates agreement no better
than chance, while an ARI of 1 indicates identical groupings. Because
several occupations overlap heavily, absolute ARI values are expected to
be low, so we compare against the same procedure applied to raw skill
counts without clusters; a taxonomy that matches or exceeds this
baseline with far fewer dimensions captures meaningful structure. We
also report the fraction of retained skills that appear in only one of
the two datasets, which quantifies the vocabulary gap noted in
Section [2](#data){reference-type="ref" reference="data"}.

## Controlled Skill Substitution {#eval:substitution}

To test partial credit directly, we replace one skill in each test
resume with (a) a skill from the same cluster (e.g., React $\rightarrow$
Vue) or (b) a skill from a different cluster (e.g., React $\rightarrow$
Java). The substitution is made in the resume's structured record, which
is then re-labeled (step 5), re-embedded, and re-ranked, so that no
other part of the document changes. We then measure how far each
original top-10 posting falls in the ranking. Only postings that mention
neither substitute are included, and substitutes are drawn from skills
present in both datasets, so that differences are not caused by the gap
between collection periods. If the taxonomy works as intended, rank
drops in Full should be consistently smaller for (a) than for (b).
Conversely, B1 should penalize both substitutions equally, since neither
substitute overlaps with the posting, while B2 is expected to fall
between the two, since embeddings capture some similarity without the
taxonomy.

## Explanation Quality {#eval:explanation}

For the top-ranked match of each test resume, the two LLM judges
(Section [5.2](#eval:judge){reference-type="ref"
reference="eval:judge"}) rate the agent's explanation from 1 to 5 on
faithfulness, meaning that every claim is supported by the resume and
posting [@maynez2020faithfulness], and on usefulness to a candidate or
recruiter. As a sanity check, both authors review 10 of these
explanations. Hallucinations are detected automatically: skills named in
each explanation are matched against the final skill sets from step 5,
and any skill that appears in neither document is counted as a
hallucination, while references to cluster names are not counted.
Misattributed skills, such as a posting requirement described as a
candidate's skill, are captured by the faithfulness rating. The
hallucination rate is reported separately.

## Statistical Analysis and Limitations {#eval:stats}

Because the test set contains only 100 resumes, we report 95% bootstrap
confidence intervals [@efron1993bootstrap] and compare configurations
with a paired randomization test on per-resume scores
[@smucker2007significance]. It is also worth noting that most labels
come from LLM judges with only a small human check, and that the judges
may share the embedding models' notion of similarity, which could favor
B2 and Full over B1; the controlled substitution test
(Section [5.5](#eval:substitution){reference-type="ref"
reference="eval:substitution"}) does not rely on these labels. B1 also
sees skills only, while B2 and Full additionally embed title and
experience level, so part of any gap between B1 and the embedding
systems may reflect seniority rather than vocabulary. Explanations are
generated after ranking and are evaluated against the documents, not
against how the similarity score was computed. Additionally, labels
measure judged relevance rather than hiring outcomes. Furthermore, the
resumes predate the postings by at least three years, and evaluation is
limited to software engineering and adjacent fields.

# Project Timeline {#logistics}

The project is planned over eight weeks
\[Table [4](#tab:timeline){reference-type="ref"
reference="tab:timeline"}\]. The build phase is completed first, as
every later step depends on the structured documents and the skill
taxonomy, and Week 8 also serves as a buffer if LLM structuring or
labeling takes longer than expected.

::: {#tab:timeline}
   Week  Tasks                                                                                                                                                                                                                                    Deliverable
  ------ ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ----------------------------------
    1    **Data:** clean both datasets; filter postings by title and industry; set up Qwen3-8B with vLLM and the Pydantic schemas                                                                                                                 Filtered subsets and their sizes
    2    **Structuring:** first-pass structuring of all filtered documents; build the skill alias table; sample the development and test resumes                                                                                                  Structured documents
    3    **Taxonomy:** embed the most frequent skills; compare clustering methods using silhouette score and noise fraction; review clusters manually and select the taxonomy                                                                     Skill taxonomy
    4    **Matching:** second-pass, cluster-aware structuring; implement the Jaccard baseline (B1); embed documents with and without cluster labels; implement two-way retrieval with NumPy for the embedding baseline (B2) and the full system   B1, B2, and Full rankings
    5    **Labeling setup:** build the FastAPI backend and React review page; pool results; set up the two LLM judges; finalize the rubric and judge prompts on the development set                                                               Judge prompts, rubric
    6    **Labeling:** label pooled pairs with both judges and compute $\kappa$; label 20 spot-check pairs; build the LangGraph agent; run the skill substitution test                                                                            All labels collected
    7    **Evaluation:** compute ranking metrics with bootstrap confidence intervals and ARI; rate explanations with the judges, review 10 by hand, and measure the hallucination rate; complete the React interface; analyze errors              Results and working application
    8    **Reporting:** write the final report; prepare the demonstration and presentation; buffer for delays                                                                                                                                     Final report and presentation

  : Eight-week project timeline.
:::

# Division of Work {#division}

As a two-person group, both authors contribute equally to every stage of
the project, including data processing, model development, the
application, evaluation, and the final report. Tasks within each week of
the timeline \[Table [4](#tab:timeline){reference-type="ref"
reference="tab:timeline"}\] are completed jointly rather than assigned
to one author. Both authors also jointly label the spot-check pairs and
review the sampled explanations used to validate the LLM judges. Code is
shared through a common repository, and progress is reviewed together at
the end of each week to adjust the timeline if needed.

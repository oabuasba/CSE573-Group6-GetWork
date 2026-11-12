# Deviations from the proposal

A living tracker of every decision in this project that departs from, or disagrees with, `docs/proposal.md`. The later presentation (`docs/Group6-Project3-FA26-Group-Project-Presentation.txt`) is a revision of the proposal and already wins where they differ (architecture ADR-1). The proposal text has not been edited; this file records what must change in it before the report.

Maintained by: the planner adds and updates rows; the reviewer checks this file.

## Rules

1. IDs (`P-n`) are never reused.
2. A planner adds or updates a row whenever an ADR, D-id, CC or Q-id changes something the proposal or presentation states.
3. The reviewer checks for a new row whenever a spec cites "the proposal".
4. Rows are closed (Status/Resolution updated), never deleted.

Types: `Departure`, `Superseded-by-presentation`, `Clarification`, `Addition`, `Open`.
Status values: `built`, `specced`, `draft`, `proposed`, `open`.

## Summary

| ID | Area | Title | Type | Status | Proposal text change needed |
|----|------|-------|------|--------|-----------------------------|
| P-1 | Structuring | Model hosting: Voyager instead of self-hosted Qwen3-8B/vLLM | Departure | built | Yes (ADR-11 lists wording; none applied) |
| P-2 | Structuring | Model actually used is `qwen38-27b`, named in no spec | Open | built (trial only) | Yes, after Q-8 closes |
| P-3 | Structuring | "Regenerate until valid" became bounded retry that can drop documents | Departure | built | Yes (one sentence) |
| P-4 | Structuring | Local LLM backend reintroduces Qwen3-8B differently | Departure | draft | Conditional |
| P-5 | LLM steps | Different LLMs may serve different LLM steps | Departure | draft | Methods must list model per step |
| P-6 | Preprocessing | Domain filter widened from software to six families | Departure | built | Yes |
| P-7 | Preprocessing | Exact duplicates became normalized near-duplicates | Clarification | built | Yes (one clause) |
| P-8 | Retrieval | B1 replaced by B1+; SciPy added | Superseded-by-presentation | specced | Yes |
| P-9 | Evaluation | Labeling and validation protocol replaced | Superseded-by-presentation | specced | Yes |
| P-10 | Evaluation | Comparisons and metrics restated | Superseded-by-presentation | specced | Yes (fold into P-9) |
| P-11 | Timeline | Weeks 1, 4, 6 shifted; unplanned stages added | Superseded-by-presentation | specced | Yes |
| P-12 | Embedding | Experience-level derivation | Clarification | draft | Yes |
| P-13 | Relabeling | LLM-free default; add-only re-extraction | Departure | draft | Yes |
| P-14 | Embedding | Optional Voyager embedding backend | Addition | draft | Conditional |
| P-15 | Web app | Query-time hosting: user-run local process with lazy, optional models | Clarification | built | Yes (one paragraph in 3.7) |
| P-16 | Structuring | Alias table: how it is built, and the display-spelling rule | Clarification | built | Yes (one or two sentences) |
| P-17 | Web app | Deterministic explainer ships before the agent; chat and blind labeling deferred; demo mode and ASU-brand theming added | Addition | built | Yes (scope sentence) |
| P-18 | Agent | Agent uses a JSON action protocol (not native tool calls), a claim verifier, plain-language errors on failure with a logged redacted cause, partly verified answers with a note, stateless non-streamed chat served only by the agent (rules-based explanations; `--explainer auto` default) | Clarification | built | Yes (one paragraph in 3.7) |
| P-19 | Agent | Agent model is served by Voyager by default, not ASU CreateAI | Clarification | built | Yes (one clause in 3.7) |

Row count: 19.

---

## P-1 Structuring model hosting

- **Proposal says:** Section 3.2 Data Structuring (~lines 149-154), Table 2 tools row, Step 2: self-host Qwen3-8B on ASU Sol GPUs (Colab fallback) with vLLM; "keeps the cost fixed and makes results reproducible"; rejects rate-limited external APIs.
- **Now:** ADR-11 supersedes ADR-3 for where the model runs and which model. An `LLMClient` protocol talks to ASU Voyager (openai SDK). The model id has no default (structurer D-8). A model other than Qwen3-8B is acceptable (ADR-11 point 4). Records store `model`, `prompt_version`, `schema_version` (D-7).
- **Reason recorded:** Voyager is free on ASU hardware, Sol GPU access is not arranged, and Voyager lists no Qwen3-8B.
- **Impact:** Table 2 "Qwen3-8B, vLLM, Pydantic", the Data Structuring paragraph, Step 2, Timeline week 1, presentation Tools slide. Reproducibility now rests on the logged model id.
- **Status:** built (structurer).
- **Proposal text change:** Needed. ADR-11 lists the wording; none applied.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open

## P-2 Model actually used is `qwen38-27b`

- **Proposal says:** Qwen3-8B (see P-1).
- **Now:** Trial summaries in `data/structured_trial/{resumes,postings}/*_summary.json` record model `qwen38-27b`, prompt `extract_v1`, schema_version 1, reasoning_effort none, temperature 0.0, 100 records each (seed 0). Q-8 (which Voyager model id) is still open in architecture.html and no ADR names the model. The trial used `--no-response-schema` (use_response_schema false) because the preflight reported `supports_response_schema` false for this model; structurer D-9 default is strict schema on. Size 27B is inferred from the id (unverified).
- **Reason recorded:** Only in run summaries (preflight result for the schema flag).
- **Impact:** Same places as P-1; the final model must be stated in the report.
- **Status:** built (trial only).
- **Proposal text change:** Fill in once the final model is chosen.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** Close Q-8 with an ADR.

## P-3 Regenerate-on-failure became bounded retry that can drop documents

- **Proposal says:** (~lines 157-159) invalid output is regenerated, so every document follows the same structure.
- **Now:** At most 3 model calls per record with the validation error fed back (D-14). Outlier validators (D-21): skill over 100 chars, over 200 skills, over 50 experience or over 20 education entries. Failed ids go to a failures CSV and are retried on the next run (D-10). Observed: resume trial had 1/100 permanent failure (r-09272, "each skill must be at most 100 characters", 3 attempts).
- **Reason recorded:** Bounded cost and refusal to silently truncate outliers (D-21).
- **Impact:** No spec says what downstream stages do with ids that never succeed. A reported failure count and a policy are needed.
- **Status:** built.
- **Proposal text change:** Mild, one sentence.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open (policy for permanent failures missing)

## P-4 Local LLM backend reintroduces Qwen3-8B differently

- **Proposal says:** Qwen3-8B on Sol/Colab with vLLM, full precision.
- **Now:** ADR-13 / local-llm-backend (draft): the same adapter serves a local OpenAI-compatible server (vLLM, Ollama or llama.cpp) on the user's RTX 5070 (12 GB). Recommended: Qwen3-8B 4-bit (AWQ/GGUF), because full precision (~16 GB) does not fit (L-7). Differences: quantized weights, not Sol/Colab, possibly Ollama/llama.cpp instead of vLLM, `--max-input-chars 24000` vs default 48000 (more truncation), 2 workers, response schema off by default (L-5), thinking-off is server-specific (L-6). `backend` joins the provenance tuple so local and Voyager data cannot be mixed (`ProvenanceMismatchError`).
- **Reason recorded:** Local GPU is available; Sol is not.
- **Impact:** If the final dataset comes from the local backend, the report must say "Qwen3-8B 4-bit on a local GPU, truncation limit X".
- **Status:** draft.
- **Proposal text change:** Conditional on which backend produces the final data.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open

## P-5 Different LLMs may be used per LLM step

- **Proposal says:** Implies one model for both passes, reasoning disabled for both.
- **Now (draft):** cluster-relabeling's relabel LLM "need not be the structuring model"; alias-table has an LLM proposer; skill-taxonomy has an optional LLM cluster namer; all built via the `make_client` factory (ADR-13).
- **Reason recorded:** Flexibility per step.
- **Impact:** Mitigation: relabel caches carry `taxonomy_version`, `model`, `backend`, `prompt_version`; mixing raises `RelabelProvenanceError`. Methods must list the model per LLM step. Judges must not match any pipeline model (proposal 5.2).
- **Status:** draft.
- **Proposal text change:** Methods section lists model per step.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open

## P-6 Domain filter widened from "software-related" to six families

- **Proposal says:** Step 1 / 3.1: keep postings whose title matches software-related terms, industry as a secondary check. Presentation: "keep SWE postings".
- **Now (posting-preprocessing, done):** families software, database, sysadmin, network, security, project_manager (incl. program manager, scrum master). Title-first with exclusions (sales, recruiter, real estate, etc.). Industry is used only for project managers (they need a tech title word or tech industry), because 2,204 software-titled postings sit in non-tech industries.
- **Reason recorded:** The ten resume occupations include DBAs, sysadmins, network admins, security analysts and PMs.
- **Impact:** Built result 8,423 kept: software 5,260; sysadmin 1,105; PM 933; security 680; network 323; database 231 (multi-family postings counted in each). Affects Step 1 wording and the presentation "SWE postings"/"Pool" line. The 5,000-posting evaluation cap binds. Answers the proposal's "size of each software engineering subset" sentence.
- **Status:** built; the spec has a "Please confirm" for assumption 1.
- **Proposal text change:** Yes (Step 1, 3.1).
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open (awaiting confirmation of assumption 1)

## P-7 "Exact duplicates" became normalized near-duplicates

- **Proposal says:** Exact duplicates are removed.
- **Now:** Resumes: dedup after case/whitespace normalization of cleaned text (770 dropped), keep the smaller id, merge labels as an ordered union. Postings: keyed on (title, description) casefolded, ignoring company: 638 dropped, smallest job_id kept. Other thresholds pinned: min_words 50, alpha ratio 0.5, control ratio 0.01, language prob 0.90; resumes dropped 2 non-English and 169 too short.
- **Reason recorded:** Pinned in preprocessing specs.
- **Impact:** Cleaning description in the report.
- **Status:** built.
- **Proposal text change:** One clause.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open

## P-8 B1 (skills-only Jaccard) replaced by B1+ plus SciPy

- **Proposal says:** 3.5, Step 7, 5.3: B1 = Jaccard of final skill sets; tools: Retrieval = NumPy.
- **Now:** ADR-1 / ADR-8 (accepted, from the presentation): B1+ = skill Jaccard + title overlap + seniority match, weights tuned on the dev set, so comparisons change one factor. Draft details (two-way-retrieval RT-3..RT-5, Q-2): token Jaccard on titles with stop-words; ordinal level distance, 0.5 if unknown; simplex grid search; default weights 0.7/0.15/0.15 until tuned; SciPy sparse (RT-4).
- **Reason recorded:** Matched-input comparisons.
- **Impact:** Proposal changes: Table 2 Retrieval "+ SciPy", Step 7, 5.3, 5.6 (the "B1 sees skills only" limitation is gone).
- **Status:** accepted, not built; formula draft (Q-2 open).
- **Proposal text change:** Yes.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open (Q-2)

## P-9 Labeling and validation protocol replaced

- **Proposal says:** 5.1/5.2: dev 10 resumes; test 100 resumes + 20 postings for the reverse direction; human check of 20 consensus pairs (4 resumes x 5 postings); relabel if kappa < 0.6.
- **Now (ADR-1 / ADR-9):** dev 10 resumes + 10 postings; test 100 + 100; human check 10+10 from the test set; ~400 pooled top-10 pairs labeled independently with no adjudication; judge as third annotator. Reported: author-author kappa, each judge vs each author, Kendall tau, metric gaps. Rubric frozen after dev with a one-time revise-on-half fallback; judge-count and lower-label rules unchanged.
- **Reason recorded:** Presentation revision.
- **Impact:** 5.1, 5.2, 5.6, Table 4 weeks 4-6, hours estimate (~7 h per author).
- **Status:** specced (ADR-9); evaluation features proposed with no spec; Q-5 (judge models/access) open.
- **Proposal text change:** Yes.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open

## P-10 Evaluation comparisons and metrics restated

- **Proposal says:** Configurations B1, B2, Full; paired randomization test on per-resume scores.
- **Now (ADR-1):** B1+, B2, Full; per-direction metrics; results under judge labels and under each author's labels, with "Full > B2 > B1+ under both authors" expected; randomization test on per-query scores using judge labels.
- **Reason recorded:** Presentation revision.
- **Impact:** Table 3, 5.3. Fold text changes into P-9.
- **Status:** specced.
- **Proposal text change:** Yes (with P-9).
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open

## P-11 Timeline shifted

- **Proposal says:** Week 1 "set up Qwen3-8B with vLLM"; weeks 4 and 6 per original plan.
- **Now:** Week 1 is a Voyager structuring script. Week 4 produces matched-input variants (B1+). Week 6 carries posting-side judging and ~400 human pairs. The build also added unplanned stages and a trial-first workflow (ADR-12).
- **Reason recorded:** Follows P-1, P-8, P-9.
- **Impact:** Table 4.
- **Status:** specced.
- **Proposal text change:** Yes.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open

## P-12 Experience level

- **Proposal says:** 3.3: postings take the listed field; resumes derive from total years.
- **Now (draft, document-embedding DE-3, answering Q-3):** LinkedIn six-level scale + "Not specified". Postings use the listed level, else (28.6% missing in the candidate set; 2,375 of 8,423) derive from `years_experience`, else "Not specified". Resumes always derive: under 2 years Entry level, 2 to under 5 Associate, 5+ Mid-Senior (`--level-bands`). Director, Executive, Internship are never derived, so resume-side seniority is asymmetric with posting-side.
- **Reason recorded:** Fill the 28.6% gap and give resumes a level.
- **Impact:** 3.3, B1+ seniority, seniority-mismatch cap on labels (CC-7).
- **Status:** draft; Q-3 open with a proposed answer.
- **Proposal text change:** Yes.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open (Q-3)

## P-13 Cluster relabel default is LLM-free; re-extraction is add-only

- **Proposal says:** Step 5: unseen skills "are passed to the LLM"; incomplete documents are re-processed from raw text.
- **Now (draft, cluster-relabeling RL-2, RL-5):** LLM is opt-in (`--use-llm`). Default `lookup_only` sends unseen skills to Other (via "default") and only flags incomplete documents. Re-extraction sends cleaned text (cut to 48,000 chars), known skills and cluster names, asks for additional skills only (max 50), and never removes pass-1 skills. `relabel_mode` is stamped into artifacts and sources must match.
- **Reason recorded:** Reproducible, cost-free default.
- **Impact:** The report must state which mode produced results.
- **Status:** draft.
- **Proposal text change:** Yes (Step 5).
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open

## P-14 Embedding backend: optional Voyager route

- **Proposal says:** Qwen3-Embedding-4B via sentence-transformers, one model for both spaces.
- **Now (draft, ADR-14 / CC-13 / embedding-client):** default is sentence-transformers on a local GPU (CUDA only, fp16, max_seq_length 2048, batch 8 with OOM halving). Voyager is a second backend (model id from `VOYAGER_EMBED_MODEL`, no default). Vector sets from different backends/models are refused. No instruction prefix on any text. 8B is out of scope.
- **Reason recorded:** Optional alternative if local GPU is insufficient.
- **Impact:** Report should state dtype, sequence limit and no prefix.
- **Status:** draft (conditional).
- **Proposal text change:** Conditional on Voyager route being used.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** open

## P-15 Query-time hosting of structuring and embedding

- **Proposal says:** 3.7: submitted documents are structured by an LLM call, then relabeled and embedded.
- **Now (built, web-app, ADR-15; answers Q-1):** the app is one user-run local process (`webapp serve`) that reads the user's own `.env` through the CLI only and builds the structuring client (`make_client`) and the embedding client (`make_embedding_client`) lazily. Query kinds degrade: a corpus document needs neither model, a form needs only the embedding model (nothing for B1+), free text needs both. New documents are labeled lookup-only (unseen skills go to Other), not with the LLM relabeler. Nothing from a query is stored or logged.
- **Reason recorded:** CC-10 forbids agents holding the key and Voyager models may be retired; Sol is a batch cluster; the 12 GB card cannot hold a local LLM server and the embedding model together.
- **Impact:** A live demo needs a configured team machine. The proposal's "relabeled" step at query time is lookup-only unless the LLM relabeler is wired in later.
- **Status:** built (web-app, 2026-11-11).
- **Proposal text change:** One paragraph in 3.7 describing the local, optional, lazy model use and the lookup-only labeling of new documents.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** built; Q-1 answered in ADR-15; the proposal paragraph in 3.7 is still to be written

## P-16 Alias table construction and display spelling

- **Proposal says:** Step 2 (~lines 281-283): extracted skills are normalized with an alias table that maps variants to a canonical form (JS to JavaScript, k8s to Kubernetes). It does not say how the table is made.
- **Now (built, alias-table):** two layers. (1) Every skill spelling in the structured records of both sources is grouped by a normalized key (NFKC, whitespace collapse, casefold, trailing `.,;:` stripped) and the group takes one display spelling; (2) a curated CSV (`src/getwork/alias/alias_curated.csv`, 38 seed rows written by the planner, not mined from data) overrides it. Matching is exact on the key; no fuzzy or embedding matching. An optional, user-run LLM step only drafts rows for human review and is never merged automatically. Every record keeps `skills_raw` and an `alias_table_version`; the trial and full-corpus tables differ, so the version must match across both sources downstream.
- **Display-spelling rule (A-3):** ignore all-caps forms of 5 or more letters when another form exists, then most frequent, then not-all-lower-case, then the smaller string. Spec decision (2026-11-08): the threshold is "more than 4 letters" because on the trial data a 4-letter threshold would have picked `Html`, `Rest`, `json`, `Soap` over `HTML`, `REST`, `JSON`, `SOAP`. Built as `SHOUTING_MIN_LETTERS = 5` in `src/getwork/alias/build.py`; `test_pick_display_rules` asserts `JAVA` over `Java` and `HTML` over `Html`.
- **Follow-up closed (2026-11-08):** the first built code used a threshold of 4 because the first spec example contradicted the prose; it was fixed to 5 and the test updated. Any alias table built before the fix has a different version, so rerun `alias-table build` and `apply` and rebuild downstream artifacts (CC-12 refuses mixed versions); not verified here whether the trial artifacts were already rebuilt.
- **Reason recorded:** The proposal gives examples only; a curated-only table cannot cover tens of thousands of skills, and case variants are 208 of 2,059 distinct trial resume skills.
- **Impact:** Methods text for Step 2; the report should say the table is corpus-derived plus a small curated list, and quantify merged forms from `alias_build_report.json`.
- **Status:** built.
- **Proposal text change:** One or two sentences in Step 2.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** closed in code; the proposal text change below is still open.

---

## P-17 Web app scope and presentation

- **Proposal says:** Application row and 3.7: "FastAPI backend, React frontend allows users to submit a resume or job", view top-k, chat with the agent, and label pairs blind (presentation slide: "React app: submit a resume or posting, view top-k, chat, and label pairs blind").
- **Now (built, web-app, ADR-16 to ADR-18, ADR-28):** FastAPI and React as stated. The explanation shown with each match is deterministic evidence (shared, related by cluster, missing skills) plus a rule-based `TemplateExplainer` behind an `Explainer` protocol; the LangGraph agent (`match-agent`) plugs into that seam later, and chat follow-ups go with it. Blind labeling becomes its own feature (`labeling-ui`) after `eval-sampling-pooling`. Added: a demo mode on built-in data, a corpus-document query kind, and a UI theme built from the public ASU brand-guide values (text wordmark, no ASU marks, no ASU repository code).
- **Reason recorded:** the agent and the labeling inputs do not exist yet; a faithful, testable explanation can ship now and doubles as a baseline for the agent's hallucination rate.
- **Impact:** Methods text for the application; the report must not claim the shipped UI explanations come from the agent.
- **Status:** built (web-app, 2026-11-11). Addition: the artifacts location is changeable (`--artifacts-dir`, `WEBAPP_ARTIFACTS_DIR`) and may be a folder of `vN` runs (ADR-28).
- **Proposal text change:** One scope sentence in 3.7.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** built; the scope sentence in the proposal is still to be written

## P-18 Agent implementation: protocol, verification, failure behavior

- **Proposal says:** A LangGraph agent with tools (similarity search, cluster lookup, skill comparison) explains each match, reached through ASU CreateAI; explanations are rated 1 to 5 for faithfulness and usefulness, and a hallucination is a skill named in neither document.
- **Now (built, match-agent, ADR-19, ADR-20, CC-19, ADR-27, ADR-30, ADR-31):** LangGraph is used as the state machine only (optional `agent` extra, imported lazily), kept because the proposal names it rather than because a four-node loop needs it. The model is called through the shared `LLMClient`, so tool use is a JSON action protocol, not native function calling. Four read-only tools: document skills, compare two results, cluster lookup (over skills already in the conversation), and similar documents. Every answer is a list of claims citing code-generated facts, and a deterministic verifier rejects any claim that names a skill outside the cited documents. If the model fails, times out (60 s), exhausts its budget (4 tool steps, 6 model calls) or stays unverified after two retries, the user sees a short plain-language error with a suggestion (HTTP 502, error code `assistant_failed`) and no explanation text; there is no automatic fallback to the rules-based explainer for a failed chat answer. Failures are diagnosable: each failed turn logs one redacted `agent failed: reason=<r> detail=<cause>` line on logger `getwork.agent` (the `webapp serve` path installs a stderr handler for the `getwork` logger), HTTP 400/413/422 from the model service map to reason `model_rejected` (previously `llm_unavailable`), and the 502 error body carries `error.reason`. Verification degrades gracefully: after the retries the verified claims are kept and the rest dropped (`ChatResponse.note`, `AgentResult.dropped_claims`); `unverified` is raised only when no claim survives. Lower-case plain English words that collide with vocabulary entries (`COMMON_WORD_KEYS`, a short list such as lists, code, tools, support; never words that can be skills like cloud or security) are not counted as skill mentions. Prompt is `agent_v2` (improvement questions answered from gap_skills). The `unverified` detail may name the rejected skill (a deliberate exception to "no skill text in logs"). Since 2026-11-11 the agent serves only `POST /api/chat`: `POST /api/explain` is always the rules-based `TemplateExplainer` (no model call, regardless of `--explainer`), and `/api/health` reports the agent as `chat` (`ChatInfo`: `enabled`, `model`, `backend`, `prompt_version`, `reason`) while `explainer` is always `template`. `--explainer auto` (default) enables chat when the LLM settings and langgraph resolve and otherwise starts with chat off and a stderr notice (`--explainer agent` makes the same problem an exit 2; `--explainer template` turns chat off quietly). Known, accepted limits of the console log redaction: a secret straddling character 200 of the logged reply excerpt, an upper-case `HTTPS://` URL, and authorization header forms other than `Bearer` are not caught; the console is the user's own terminal. Chat is stateless and not streamed. As built: the call budget (6) caps decisions and parse attempts but not the shared client's transient-error retries (up to 5 calls per request), and on an unexpected bug the web-app 500 handler logs the exception text locally (ADR-27).
- **Reason recorded:** `LLMClient` has no tool-call support and local servers differ; verified output is what makes the hallucination rate meaningful and keeps the demo safe; the user chose visible errors over a silent change of explanation type.
- **Impact:** Methods text for step 8. Because the verifier rejects ungrounded skills before they are shown, the automatic hallucination rate on accepted answers is near zero by construction; the report must also state the verifier intervention rate, the error share by failure reason, and the human or judge faithfulness ratings, and must say the vocabulary scan cannot see skills outside the taxonomy and alias table, nor skills spelled with a hyphen or slash (Front-end, scikit-learn, CI/CD; ADR-27). The agent model must differ from both judge models (ADR-10); this is a configuration rule, not enforced in code.
- **Status:** built (match-agent, reviewed 2026-11-11).
- **Proposal text change:** One paragraph in 3.7.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** built; report sentences and the proposal paragraph in 3.7 still to be written

## P-19 Agent model hosting: Voyager instead of CreateAI

- **Proposal says:** The explanation agent runs on LangGraph and its model is reached through ASU CreateAI (ADR-10).
- **Now (built, match-agent, ADR-21):** The agent uses the same Voyager backend and model as structuring by default. `--agent-backend local` (with `LOCAL_LLM_BASE_URL`) still reaches any OpenAI-compatible server, CreateAI included. Only where the API is served changes; the agent design, tools and the rule that the agent model differs from both judge models are unchanged.
- **Reason recorded:** the user chose to serve the agent where the project's other LLM calls already go; the shared client has no CreateAI backend and its endpoint and authentication were not confirmed.
- **Impact:** Methods text that names CreateAI; the report must name the model and host the agent actually used.
- **Status:** built (match-agent, reviewed 2026-11-11).
- **Proposal text change:** One clause in 3.7.
- **Report sentence:** TBD
- **Owner:** TBD
- **Resolution:** built; the report must name the model and host actually used

## Gap-fillers and additions (not disagreements)

- `project-foundation` (repo layout and shared config) was cancelled by the user: the layout grew feature by feature, and the CC-1 models, id scheme and model client came with `structurer`, `alias-table` and `local-llm-backend`. The proposed repo conventions in the architecture (Q-6) stay unconfirmed.
- Structurer: `experience[]` = {role, company, duration_months}, `education[]` = {degree, field, institution}; all keys required, values nullable (D-1); role nullable.
- No experience-level field from the structurer; numeric `years_experience` (resume: explicit total, else sum of durations/12; posting: minimum years required); level left to document-embedding (D-3, D-2).
- Skills stored as written with only trim and case-insensitive dedup (D-4); alias table is a separate step (CC-2).
- Same `ModelOutput` schema for postings and resumes (D-2).
- Input truncation at 48,000 chars (D-15; 3 of 28,842 resumes, 0 postings); 4 workers.
- Provenance refusal on mixed model/prompt/schema (D-7/D-10/D-16/D-19/D-24/D-25).
- Downstream posting title is the LLM-extracted title, not the raw CSV field.
- Alias-table (built): corpus-built by case/whitespace key plus a curated CSV; optional LLM proposer writes a draft CSV that is never auto-merged; exact normalized-key matching; `skills_raw` retained (see P-16 for the display-spelling rule).
- Alias-table implementation details not in the proposal or the first spec text, now written into the spec: `load_table` raises `ValueError` for a file that is not an object with an `entries` object; `build`, `apply` and `propose` pre-validate every structured record (all `RECORD_KEYS`, list `skills`) and exit 2 before writing anything; `propose` calls `make_client` before reading any input; the proposer's `conflict` status is unreachable in real runs (each skill appears once in `canonical_counts`, and `parse_groups` rejects an alias in two groups), so it is a defensive guard tested with synthetic input; `ModelOutput` dedupes skills case-insensitively, so tests with case-variant raw skills overwrite `record["skills"]` after `to_record`; the `alias-table` script entry is added in T-1, so `uv run alias-table` fails until T-8; AC-21 runs `uv run --offline pytest`.
- Alias-table style note: spec signatures `complete_validated` and `propose_aliases` have three or four positional parameters then keyword-only ones; `call_with_retry` (six positional) is the structurer's existing function, called unchanged. The project's keyword-style rule applies to new functions in later features.
- Skill-taxonomy (draft): document-frequency counting, min_df 2; UMAP 5 components / 15 neighbors / min_dist 0 / cosine; default grids; noise cap 0.30; silhouette on original vectors, clustered skills only; k-means also on UMAP points; editable `cluster_names.csv`; heuristic naming from the three most frequent members; names at most 40 chars, unique, not "Other"; hand editing limited to renaming or choosing a configuration.
- Taxonomy cluster naming (built; ADR-23 to ADR-26): the optional LLM cluster namer (`taxonomy name --method llm`) is followed by two bounded LLM review passes, a fit review (ADR-26, `--fit-rounds`, `NAME_FIT_ROUNDS`, default 1) and an overlap review (ADR-25, `--overlap-rounds`), both on by default in that mode. They add Voyager calls: one judge call per round plus one to three calls per flagged cluster per round (at most 3 validation attempts); `--fit-rounds` is ignored for `--method heuristic`. Unresolved names get status `weak_fit` or `overlap`; `taxonomy build` only warns. The proposal does not mention cluster naming or its review; the report methods text should say names are model-written and reviewed by a model, and that the model check is not a human validation.
- Web UI changes requested directly by the user on 2026-11-11, without a prior spec (now written into the web-app and match-agent specs): the "What we understood" card is a closed-by-default disclosure; the About page no longer shows data in use (counts, artifacts directory, versions, embedding identity) and describes each ranking method and the skill clusters instead, with demo mode signalled only by the global banner; with the agent explainer the result card shows a brief summary, the other points behind a toggle, and the chat opens on demand. None of these is in the proposal. A candidate name or contact idea is deferred.
- Relabeling: "incomplete" = `no_skills`, `few_skills` (below the per-source 10th percentile, Q-9) or `low_coverage` (<0.5 known); lookup always wins (RL-1); unseen skills resolved once per distinct skill in batches of 50, with caching.
- Document serialization: labeled lines "Title: / Level: / Skills: Cluster A: s1, s2; Other: ...". B2 order choice open as Q-10 (the grouped default lets adjacency leak taxonomy information to B2, an ablation-validity issue).
- Retrieval: exact blockwise top-k, k=10, six Parquet ranking files (CC-17).
- Data facts: 745 of kept resumes have no occupation label (the proposal says every resume has one or more of the 10 labels), so stratified sampling and ARI need a rule for unlabeled resumes. Sizes: 123,849 postings (not ~124,000); 28,842 resumes kept of 29,783.
- Engineering conventions: uv, `src/getwork` layout, trial-first defaults (CC-11, ADR-12), content-version stamps (CC-12), `artifacts_trial/` git-ignored.

## Implied or undocumented

1. `qwen38-27b` appears only in run summaries; Q-8 open, no ADR (see P-2).
2. The trial ran with response schema off; the reason is in no spec (it is the preflight result; see P-2).
3. `.gitignore` excludes `data/job_postings/` and `data/resume_corpus/`, while architecture (Q-7) says raw data is committed because the instructor requires it. The repo contradicts the stated convention.
4. Built resume ids are `r-<file stem>` (e.g. `r-09272`); the architecture convention says `r-<row>`.
5. No downstream policy for documents that fail structuring (P-3).
6. Judge independence: the proposal says neither judge is a pipeline or agent model; the pipeline now has several LLM steps, each possibly on a different model, and ADR-11 only addresses structuring. Q-5 open.
7. ADR-13 and ADR-14 lack a "proposal wording that no longer matches" list like ADR-11's.
8. Prompt versions, quantization and server version for local runs are not captured in records (L-8 says record in project notes).
9. The `evaluations/` directory is empty with an undefined purpose.
10. Kept items with null `experience_level` or an empty label are handled by assumption, not by a recorded decision.
11. The 38-row curated alias seed is planner-written from general knowledge, not mined from the data; coverage on the full corpus is unmeasured (alias-table assumption).
12. All-caps words of 5 or more letters that are real acronyms (for example `ANSYS`) lose to a mixed-case variant under the display-spelling rule (P-16) until curated by hand.

## Pending proposal edits

- [ ] ADR-11 wording list (P-1)
- [ ] P-6 domain filter
- [ ] P-8 B1+ and SciPy
- [ ] P-9 / P-10 evaluation protocol and comparisons
- [ ] P-11 timeline
- [ ] P-12 experience level
- [ ] P-16 alias table method

## Could not verify

- Whether the fit review (ADR-26) finds the weak names it was designed for on the real corpus: the user-run checklist step AV-7 was not run by any agent (tests use a fake client only).
- Whether `qwen38-27b` is a 27B Qwen3-family model (inferred from the id) and whether it honors `reasoning_effort` none.
- The final full-corpus structuring model (Q-8).
- That the draft specs were read at a single snapshot while being edited.
- Whether ADR-12..14 and CC-11..17 are uncommitted additions.
- Why `docs/proposal.md` has not been edited.

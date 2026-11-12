# agent (pipeline step 8)

A small LangGraph agent that answers follow-up questions about a match (the per-match chat). The match card's own explanation (`POST /api/explain`) is always the rules-based one and never calls the model. It asks the
model to call one of four read-only tools or to give an answer made of claims. Every claim is checked by
a deterministic verifier against the facts the tools produced and the final skill sets of the two
documents. A claim that cannot be verified goes back to the model with the reason; if the answer still
cannot be verified, the user sees a short plain-language chat error and no answer text. The rules-based
explainer is never used as a replacement for a failed chat answer. See `specs/features/match-agent.html`.

Code: `models.py` (claims, facts, results), `facts.py`, `tools.py`, `grounding.py` (the verifier),
`prompt.py` and `prompts/agent_v2.md` (`agent_v1.md` is kept for history), `turn.py` (node logic), `graph.py` (LangGraph wiring),
`explainer.py` (explainer and chat service), `api.py` (`POST /api/chat`, the failure handler).

Tests never call a model: they use `FakeLLMClient` with scripted replies and the demo engine.

## MA-1 Install the extra and start the demo

```
uv sync --extra agent
uv run webapp serve --demo
```

The agent is the default (`--explainer auto`). It uses the same LLM settings and model as the structurer:
`VOYAGER_API_KEY`, `VOYAGER_BASE_URL` and `VOYAGER_MODEL` (or `LLM_BACKEND=local` with `LOCAL_LLM_*`) from `.env` or the
environment, so `--model` is not needed when `VOYAGER_MODEL` is set; `--backend` and `--model` override them. User-run
step: it makes real model calls with your key. If a setting or `langgraph` is missing, the server starts with chat off,
prints one `chat agent disabled: ...` notice naming the missing variables, has no chat route, and
`/api/health` shows `chat.reason` (`chat.enabled` is false). Use `--explainer agent` to make that an error (exit 2) and
`--explainer template` to turn chat off.

With real artifacts, replace `--demo` by `--artifacts-dir artifacts_trial` (or `WEBAPP_ARTIFACTS_DIR`); the agent's skill
vocabulary is read from the run folder the server resolved (for example `artifacts_trial/v5`), and the page works the
same way.

Pass: `uv run webapp check --demo` prints health with `"chat": {"enabled": true, ...}` (or `enabled: false` and a `reason`)
and makes no model call; the server prints the address it serves on.

## MA-2 Read an explanation

Open the page, search from a demo example, expand a match ("Why this match?").

Pass: the explanation is the rules-based one (summary and bullets, "Explained by rules, no language model" line) and no model call is made. Chat
replies (MA-3) are the claim lists whose skills are all in one of the two documents; "Show evidence" lists the cited facts.

## MA-3 Ask a follow-up

In the expanded card ask "why is #1 ranked above #2", then open "Show evidence" on a claim.

Pass: the reply is made of claims, each with cited facts, and the suggested questions appear
below it.

## MA-4 Force a failure

Stop the model service, or start with a wrong model name, then send a chat message (expanding a match still works: it is rules-based).

Pass: the chat shows the plain message and hint ("The assistant could not be reached right now." or the
matching pair); the match explanation is unaffected. Restart with `--explainer template` to remove the chat button.

### Partly verified answers

When the answer still fails verification after the retries, the claims that did verify are kept and the failing ones are
dropped; the response then carries `note` ("Part of the answer was left out because it could not be checked against these
two documents.") and the chat shows it under the claims. Only when no claim survives does the turn fail with `unverified`.
Nothing unverified is ever shown. `AgentResult.as_record()` has `dropped_claims` for evaluation. Skills that are also plain
English words ("lists", "code", "tools", "support", "service", "email", and a few more; never a word that can be a skill by itself, such as cloud or security) written in lower case in prose do not count as skill mentions (a declared
skill is still checked, and capitalised or multi-word forms are still detected). Improvement questions are answered from the
match's `gap_skills` fact (prompt `agent_v2`).

### Reading a failure in the console

Every failed chat turn logs exactly one line on stderr (logger `getwork.agent`, WARNING), and the 502 body carries
`error.reason`:

```
2026-11-11 21:30:00,123 WARNING getwork.agent: agent failed: reason=model_rejected detail=PermanentLLMError status=400: ...
```

`reason` is one of: `llm_unavailable` (connection, timeout, 5xx or rate limit after retries; detail names the error class and
status), `model_rejected` (the service refused the request: 400/413/422 such as an unsupported `reasoning_effort` or
`--extra-body`, or 401/403/404/410 for a bad key, wrong base URL or wrong model; detail has the class, status and the
service's message), `invalid_reply` (the model's answer was not the expected JSON; detail has the parse error and the first
200 characters of the reply), `unverified` (no claim survived verification; detail lists the violations),
`timeout`, `budget_exhausted` (no detail). The detail is redacted: the API key, bearer tokens, key-like tokens and whole
URLs are removed, and it is truncated. The prompt and your message are never logged.

## MA-5 Reading the data in a later evaluation

`AgentExplainer.explain_detailed(...)` returns an `AgentResult`; `as_record()` is one JSON line with the
keys status, failure_reason, off_topic, claims, facts, trace, usage, model, backend and prompt_version.
`grounding.ungrounded_skills(claims, docs, vocab)` lists the skills a claim names that are in neither
document (the hallucination definition); `named_skills` gives the denominator.

Caveats the report must state: give the error share (failed results over all attempts, by
`failure_reason`) next to the hallucination rate, because failed results have no claims. The vocabulary
scan only recognises skills in the taxonomy or alias table.

## MA-6 Configuration rules

- Backend and model: by default the agent uses the structuring backend (Voyager) and model (ADR-21).
  `--agent-backend` and `--agent-model` override them; `--agent-timeout` sets the wall-clock limit.
- `--agent-backend local` reads `LOCAL_LLM_BASE_URL` (and `LOCAL_LLM_MODEL`, `LOCAL_LLM_API_KEY`) and
  works with any OpenAI-compatible server, including CreateAI.
- Voyager settings come from `VOYAGER_API_KEY`, `VOYAGER_BASE_URL`, `VOYAGER_MODEL` and `LLM_BACKEND`
  in the environment or `.env`; this document lists names only.
- The agent model must differ from both judge models (ADR-10). This is a configuration rule, not checked
  in code.
- On a 12 GB card do not run a local LLM server and the local embedding model together.
- What is sent to the model: the structured fields only (titles, levels, skills with their clusters, the
  result list with ranks and scores, the conversation and the question). Raw resume and posting text is
  never sent.

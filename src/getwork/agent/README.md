# agent (pipeline step 8)

A small LangGraph agent that answers follow-up questions about a match (the per-match chat). The match card's own explanation (`POST /api/explain`) is always the rules-based one and never calls the model. The model answers in plain text and may call one of four read-only tools first; there is no verifier, so the reply is shown as the model wrote it (the page says so in a disclaimer). A turn fails only for one of five reasons (see "Reading a failure in the console") and then the user sees a short plain-language chat error and no answer text. The rules-based explainer is never used as a replacement for a failed chat answer. See `specs/features/chat-conversational.html`.

Code: `models.py` (facts, results), `facts.py`, `tools.py`, `grounding.py` (vocabulary scan for the log), `prompt.py` and
`prompts/agent_v3.md` (`agent_v1.md` and `agent_v2.md` are kept for history), `turn.py` (node logic), `graph.py` (LangGraph wiring),
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
replies (MA-3) are conversational plain text.

## MA-3 Ask a follow-up

Real model, run by you (user-run; nothing in the test suite does this). Start `uv run webapp serve --demo` with real model settings, search, open "Chat about this suggestion" and ask "why is this posting ranked here" and "what should I learn".

Pass: after "The assistant is thinking…" a conversational plain-text answer appears in one piece; the greeting is at the top; there are no suggestion chips and no evidence buttons; the disclaimer ("The assistant is an AI and can be wrong...") is visible under the composer; the console shows no 502 for an answer; the match card's "Why this match?" text is unchanged.

## MA-4 Force a failure

Stop the model service, or start with a wrong model name, then send a chat message (expanding a match still works: it is rules-based).

Pass: the chat shows the plain message and hint ("The assistant could not be reached right now." or the
matching pair); the match explanation is unaffected. Restart with `--explainer template` to remove the chat button.

### Skills named outside the documents

When a reply names a skill from the taxonomy that is in neither document, the server logs one INFO line
(`agent reply names skills outside these documents: ...`, logger `getwork.agent`) and `AgentResult.as_record()` lists them in
`ungrounded_skills`. This never blocks or changes a reply. Skills that are also plain English words ("lists", "code", "tools",
"support", "service", "email", and a few more) written in lower case in prose are not counted. Improvement questions are
answered from the match's `gap_skills` fact (prompt `agent_v3`).

### Reading a failure in the console

Every failed chat turn logs exactly one line on stderr (logger `getwork.agent`, WARNING), and the 502 body carries
`error.reason`:

```
2026-11-11 21:30:00,123 WARNING getwork.agent: agent failed: reason=model_rejected detail=PermanentLLMError status=400: ...
```

`reason` is one of five: `llm_unavailable` (connection, timeout, 5xx or rate limit after retries; detail names the error class and
status), `model_rejected` (the service refused the request: 400/413/422 such as an unsupported `reasoning_effort` or
`--extra-body`, or 401/403/404/410 for a bad key, wrong base URL or wrong model; detail has the class, status and the
service's message), `invalid_reply` (the reply was empty or a malformed tool request after the retries; detail has the parse
error and the first 200 characters), `timeout`, `budget_exhausted` (no detail). The detail is redacted: the API key, bearer tokens, key-like tokens and whole
URLs are removed, and it is truncated. The prompt and your message are never logged.

## MA-5 Reading the data in a later evaluation

`AgentExplainer.explain_detailed(...)` returns an `AgentResult`; `as_record()` is one JSON line with the
keys status, failure_reason, reply, facts, trace, usage, model, backend, prompt_version and `ungrounded_skills`
(the skills the reply names that are in neither document; the hallucination measure, now taken after the fact).

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

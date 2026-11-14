import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ChatComposer from "../components/ChatComposer.jsx";
import ChatThread from "../components/ChatThread.jsx";
import ExplainPanel from "../components/ExplainPanel.jsx";
import GroupedSkills from "../components/GroupedSkills.jsx";
import MatchCard from "../components/MatchCard.jsx";
import Notice from "../components/Notice.jsx";
import { loadConversation, saveConversation } from "../conversationStore.js";
import { loadPanelOpen, savePanelOpen } from "../chatPanelPref.js";
import { loadResults, setReturning } from "../resultsSession.js";
import { hrefFor, ROUTE_BY_DIRECTION } from "../routes.js";
import { useChat } from "../useChat.js";
import "../theme/chat.css";
import "../theme/chat-split.css";

export const PANEL_ID = "chat-suggestion-panel";

const TITLE = "Chat about this suggestion";
const UNAVAILABLE = "Chat is not available on this server.";
const NO_SESSION =
  "Open a chat from a search result: run a search, open a result with Why this match?, then choose Chat about this suggestion.";

function StateShell({ children }) {
  const headingRef = useRef(null);
  useEffect(() => headingRef.current?.focus(), []);
  return (
    <div className="chat-page">
      <h1 tabIndex={-1} ref={headingRef}>{TITLE}</h1>
      {children}
    </div>
  );
}

function ChatDisabled({ reason, snapshot }) {
  const back = snapshot
    ? { href: hrefFor(ROUTE_BY_DIRECTION[snapshot.direction]), text: "Back to results" }
    : { href: hrefFor("jobs"), text: "Go to Find jobs" };
  return (
    <StateShell>
      <Notice kind="warning" label="Chat is unavailable.">{reason ?? UNAVAILABLE}</Notice>
      <p>
        <a className="link" href={back.href} onClick={snapshot ? () => setReturning(true) : undefined}>
          {back.text}
        </a>
      </p>
    </StateShell>
  );
}

function NoSession() {
  return (
    <StateShell>
      <p className="measure">{NO_SESSION}</p>
      <p>
        <a className="link" href={hrefFor("jobs")}>Find jobs</a>{" "}
        <a className="link" href={hrefFor("candidates")}>Find candidates</a>
      </p>
    </StateShell>
  );
}

function FocusMatch({ snapshot, match, fetchImpl, open }) {
  const [expanded, setExpanded] = useState(false);
  const { direction, response } = snapshot;
  return (
    <section id={PANEL_ID} className="chat-suggestion-panel" aria-label="The suggestion" hidden={!open}>
      <h2>This suggestion</h2>
      <MatchCard match={match} system={response.system} expanded={expanded} onToggle={() => setExpanded(!expanded)}>
        <GroupedSkills skills={match.document.skills} />
        <ExplainPanel direction={direction} query={response.query} matchId={match.id} fetchImpl={fetchImpl} />
      </MatchCard>
    </section>
  );
}

function useConversation({ snapshot, match, fetchImpl }) {
  const [initialTurns] = useState(() => loadConversation(match.id));
  const matches = useMemo(
    () => snapshot.response.matches.map((m) => ({ id: m.id, rank: m.rank, score: m.score })),
    [snapshot],
  );
  const onTurns = useCallback((turns) => saveConversation(match.id, turns), [match.id]);
  return useChat({
    direction: snapshot.direction,
    query: snapshot.response.query,
    focusId: match.id,
    matches,
    system: snapshot.response.system,
    fetchImpl,
    initialTurns,
    onTurns,
  });
}

function ChatSession({ snapshot, match, fetchImpl }) {
  const { turns, status, error, send, retry, reset } = useConversation({ snapshot, match, fetchImpl });
  const inputRef = useRef(null);
  const [panelOpen, setPanelOpen] = useState(loadPanelOpen);
  const last = turns[turns.length - 1];

  useEffect(() => {
    setReturning(true);
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    if (status === "idle" && last?.role === "assistant") inputRef.current?.focus();
  }, [status, turns.length, last]);

  function startOver() {
    reset();
    inputRef.current?.focus();
  }

  function togglePanel() {
    const next = !panelOpen;
    setPanelOpen(next);
    savePanelOpen(next);
  }

  return (
    <div className="chat-page">
      <div className="chat-header">
        <a className="link" href={hrefFor(ROUTE_BY_DIRECTION[snapshot.direction])}>Back to results</a>
        <div className="chat-title-row">
          <h1>{TITLE}</h1>
          <button
            type="button" className="btn btn-secondary chat-panel-toggle"
            aria-expanded={panelOpen} aria-controls={PANEL_ID} onClick={togglePanel}
          >
            {panelOpen ? "Hide suggestion" : "Show suggestion"}
          </button>
        </div>
      </div>
      <div className={panelOpen ? "chat-split" : "chat-split chat-split-solo"}>
        <div className="chat-column">
          <section className="chat-scroll">
            <h2 className="sr-only">Conversation</h2>
            <ChatThread
              turns={turns} status={status} error={error} onRetry={retry}
            />
            {turns.length > 0 && (
              <button type="button" className="btn btn-secondary" disabled={status === "thinking"} onClick={startOver}>
                Start over
              </button>
            )}
          </section>
          <ChatComposer onSend={send} disabled={status === "thinking"} inputRef={inputRef} />
        </div>
        <FocusMatch snapshot={snapshot} match={match} fetchImpl={fetchImpl} open={panelOpen} />
      </div>
    </div>
  );
}

export default function ChatPage({ matchId, health, fetchImpl }) {
  const [snapshot] = useState(() => loadResults());
  if (health?.chat?.enabled !== true) return <ChatDisabled reason={health?.chat?.reason} snapshot={snapshot} />;
  const match = snapshot?.response.matches.find((m) => m.id === matchId);
  if (!match) return <NoSession />;
  return <ChatSession snapshot={snapshot} match={match} fetchImpl={fetchImpl} />;
}

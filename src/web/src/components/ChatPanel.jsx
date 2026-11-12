import { useEffect, useRef, useState } from "react";
import { describeError } from "../api.js";
import { MAX_MESSAGE } from "../chatHistory.js";
import { useChat } from "../useChat.js";
import "../theme/chat.css";
import ClaimList from "./ClaimList.jsx";
import Notice from "./Notice.jsx";

function Turn({ turn }) {
  if (turn.role === "user") {
    return (
      <div className="chat-turn chat-turn-user">
        <p className="measure">{turn.content}</p>
      </div>
    );
  }
  return (
    <div className="chat-turn">
      {turn.off_topic ? (
        <p className="measure">{turn.reply}</p>
      ) : (
        <ClaimList claims={turn.claims} facts={turn.facts} />
      )}
      {turn.note && <p className="measure chat-note">{turn.note}</p>}
    </div>
  );
}

function ChatError({ error, onRetry }) {
  const { title, body, hint } = describeError(error);
  return (
    <Notice kind="error" role="alert">
      <strong>{title}.</strong> {body} {hint}{" "}
      <button type="button" className="btn btn-secondary" onClick={onRetry}>
        Try again
      </button>
    </Notice>
  );
}

export default function ChatPanel(props) {
  const { turns, status, error, send, retry } = useChat(props);
  const [draft, setDraft] = useState("");
  const inputRef = useRef(null);
  const thinking = status === "thinking";
  const last = turns[turns.length - 1];
  const suggestions = last?.role === "assistant" ? last.suggested_questions ?? [] : [];

  useEffect(() => {
    if (status === "idle" && last?.role === "assistant") inputRef.current?.focus();
  }, [status, last]);

  function submit(event) {
    event.preventDefault();
    if (!draft.trim() || thinking) return;
    send(draft.trim());
    setDraft("");
  }

  return (
    <section className="chat" aria-labelledby="chat-heading">
      <h3 id="chat-heading">Ask about this match</h3>
      <div className="chat-log" role="log" aria-live="polite">
        {turns.map((turn, i) => (
          <Turn key={i} turn={turn} />
        ))}
      </div>
      {thinking && <p role="status">Thinking...</p>}
      {status === "error" && <ChatError error={error} onRetry={retry} />}
      {suggestions.length > 0 && (
        <div className="chat-suggestions">
          {suggestions.map((question) => (
            <button
              key={question}
              type="button"
              className="btn btn-secondary"
              disabled={thinking}
              onClick={() => send(question)}
            >
              {question}
            </button>
          ))}
        </div>
      )}
      <form onSubmit={submit}>
        <div className="field">
          <label htmlFor="chat-input">Your question</label>
          <textarea
            id="chat-input"
            ref={inputRef}
            rows={2}
            maxLength={MAX_MESSAGE}
            value={draft}
            disabled={thinking}
            onChange={(event) => setDraft(event.target.value)}
          />
          <span className="chat-counter">{draft.length} / {MAX_MESSAGE}</span>
        </div>
        <button type="submit" className="btn btn-primary" disabled={!draft.trim() || thinking}>
          Send
        </button>
      </form>
    </section>
  );
}

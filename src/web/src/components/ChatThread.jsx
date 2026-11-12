import { useEffect, useRef } from "react";
import { describeError } from "../api.js";
import "../theme/chat.css";
import ClaimList from "./ClaimList.jsx";
import Notice from "./Notice.jsx";

const EMPTY_PROMPT = "Ask anything about why this suggestion fits, or what is missing.";

function Bubble({ turn }) {
  const isUser = turn.role === "user";
  return (
    <div className={`chat-bubble ${isUser ? "chat-bubble-user" : "chat-bubble-assistant"}`}>
      <span className="sr-only">{isUser ? "You: " : "Assistant: "}</span>
      {isUser ? <p className="measure">{turn.content}</p> : <AssistantBody turn={turn} />}
    </div>
  );
}

function AssistantBody({ turn }) {
  if (turn.off_topic) return <p className="measure">{turn.reply}</p>;
  return (
    <>
      <ClaimList claims={turn.claims} facts={turn.facts} />
      {turn.note && <p className="measure chat-note">{turn.note}</p>}
    </>
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

function Suggestions({ questions, disabled, onAsk }) {
  if (questions.length === 0) return null;
  return (
    <div className="chat-suggestions">
      {questions.map((question, index) => (
        <button
          key={`${index}-${question}`}
          type="button"
          className="btn btn-secondary"
          disabled={disabled}
          onClick={() => onAsk(question)}
        >
          {question}
        </button>
      ))}
    </div>
  );
}

export default function ChatThread({ turns, status, error, onRetry, onAsk, starters = [] }) {
  const endRef = useRef(null);
  const thinking = status === "thinking";
  const last = turns[turns.length - 1];
  const suggestions =
    turns.length === 0 ? starters : last.role === "assistant" ? last.suggested_questions ?? [] : [];

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: "end" });
  }, [turns, status]);

  return (
    <div className="chat-thread">
      {turns.length === 0 && <p>{EMPTY_PROMPT}</p>}
      <div className="chat-turns" role="log" aria-live="polite">
        {turns.map((turn, i) => (
          <Bubble key={i} turn={turn} />
        ))}
      </div>
      {thinking && (
        <div className="chat-bubble chat-bubble-assistant" role="status">
          The assistant is thinking…
        </div>
      )}
      {status === "error" && <ChatError error={error} onRetry={onRetry} />}
      <Suggestions questions={suggestions} disabled={thinking} onAsk={onAsk} />
      <div ref={endRef} />
    </div>
  );
}

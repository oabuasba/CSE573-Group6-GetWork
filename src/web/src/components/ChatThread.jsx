import { useEffect, useRef } from "react";
import { describeError } from "../api.js";
import "../theme/chat.css";
import Notice from "./Notice.jsx";

export const GREETING =
  "Hi! Ask me anything about this suggestion: why it fits, what is missing, or how to improve it.";

function Bubble({ turn }) {
  const isUser = turn.role === "user";
  return (
    <div className={`chat-bubble ${isUser ? "chat-bubble-user" : "chat-bubble-assistant"}`}>
      <span className="sr-only">{isUser ? "You: " : "Assistant: "}</span>
      {isUser ? (
        <p className="measure">{turn.content}</p>
      ) : (
        <p className="measure chat-reply">{turn.reply}</p>
      )}
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

export default function ChatThread({ turns, status, error, onRetry }) {
  const endRef = useRef(null);
  const thinking = status === "thinking";

  useEffect(() => {
    const scroller = endRef.current?.closest(".chat-scroll");
    if (scroller) scroller.scrollTop = scroller.scrollHeight;
    else endRef.current?.scrollIntoView?.({ block: "end" });
  }, [turns, status]);

  return (
    <div className="chat-thread">
      <div className="chat-bubble chat-bubble-assistant chat-greeting">
        <span className="sr-only">Assistant: </span>
        <p className="measure">{GREETING}</p>
      </div>
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
      <div ref={endRef} />
    </div>
  );
}

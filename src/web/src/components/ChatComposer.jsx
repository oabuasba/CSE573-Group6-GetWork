import { useCallback, useLayoutEffect, useRef, useState } from "react";
import { MAX_MESSAGE } from "../chatHistory.js";
import "../theme/chat.css";

export const MAX_ROWS = 6;
export const NEAR_LIMIT = 900;
export const AI_DISCLAIMER = "The assistant is an AI and can be wrong. Check anything important against the skills on the card.";
const IME_KEY_CODE = 229; // some browsers report Enter during composition only via keyCode

export default function ChatComposer({ onSend, disabled = false, inputRef }) {
  const [draft, setDraft] = useState("");
  const innerRef = useRef(null);
  const setRefs = useCallback(
    (node) => {
      innerRef.current = node;
      if (inputRef) inputRef.current = node;
    },
    [inputRef],
  );
  const rows = Math.min(MAX_ROWS, Math.max(1, draft.split("\n").length));

  // Rows only grow per newline; scrollHeight also covers soft-wrapped lines.
  useLayoutEffect(() => {
    const el = innerRef.current;
    if (!el) return;
    el.style.height = "auto";
    if (el.scrollHeight > 0) el.style.height = `${el.scrollHeight}px`;
  }, [draft]);

  function submit() {
    const message = draft.trim();
    if (disabled || message === "") return;
    onSend(message);
    setDraft("");
  }

  function handleKeyDown(event) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing || event.keyCode === IME_KEY_CODE) return;
    event.preventDefault();
    submit();
  }

  return (
    <form
      className="chat-composer"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      <label htmlFor="chat-input" className="sr-only">
        Your question
      </label>
      <div className="chat-composer-box">
        <textarea
          id="chat-input"
          aria-describedby="chat-counter"
          ref={setRefs}
          rows={rows}
          maxLength={MAX_MESSAGE}
          placeholder="Ask about this suggestion"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        <div className="chat-composer-bar">
          <span id="chat-counter" className={draft.length >= NEAR_LIMIT ? "chat-counter chat-counter-near" : "chat-counter"}>
            {draft.length} / {MAX_MESSAGE}
          </span>
          <button type="submit" className="btn btn-primary chat-send" aria-label="Send" disabled={disabled || draft.trim() === ""}>
            <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 19V5" />
              <path d="M5 12l7-7 7 7" />
            </svg>
          </button>
        </div>
      </div>
      <p id="chat-disclaimer" className="chat-disclaimer">
        {AI_DISCLAIMER}
      </p>
    </form>
  );
}

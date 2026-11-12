import { useState } from "react";
import { MAX_MESSAGE } from "../chatHistory.js";
import "../theme/chat.css";

export const MAX_ROWS = 6;

export default function ChatComposer({ onSend, disabled = false, inputRef }) {
  const [draft, setDraft] = useState("");
  const rows = Math.min(MAX_ROWS, Math.max(1, draft.split("\n").length));

  function submit() {
    const message = draft.trim();
    if (disabled || message === "") return;
    onSend(message);
    setDraft("");
  }

  function handleKeyDown(event) {
    if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return;
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
      <div className="chat-composer-row">
        <textarea
          id="chat-input"
          ref={inputRef}
          rows={rows}
          maxLength={MAX_MESSAGE}
          placeholder="Ask about this suggestion"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button type="submit" className="btn btn-primary" disabled={disabled || draft.trim() === ""}>
          Send
        </button>
      </div>
      <span className="chat-counter">
        {draft.length} / {MAX_MESSAGE}
      </span>
    </form>
  );
}

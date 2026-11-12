import { useId, useState } from "react";
import ChatPanel from "./ChatPanel.jsx";

export default function ChatOnDemand(chatProps) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  return (
    <div>
      <button
        type="button"
        className="btn btn-secondary"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen(!open)}
      >
        {open ? "Hide chat" : "Chat about this suggestion"}
      </button>
      <div id={panelId} hidden={!open}>
        {open && <ChatPanel {...chatProps} />}
      </div>
    </div>
  );
}

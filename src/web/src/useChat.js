import { useCallback, useRef, useState } from "react";
import { postChat } from "./api.js";
import { toHistory } from "./chatHistory.js";

// A trailing user turn means its request failed; a new message replaces it.
const withoutUnansweredTurn = (turns) => (turns.at(-1)?.role === "user" ? turns.slice(0, -1) : turns);

export function useChat({
  direction,
  query,
  focusId,
  matches,
  system,
  fetchImpl,
  initialTurns = [],
  onTurns,
}) {
  const [turns, setTurns] = useState(initialTurns);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  const turnsRef = useRef(initialTurns);
  const lastMessage = useRef(null);
  const historySnapshot = useRef([]);
  const busy = useRef(false);

  const post = useCallback(
    (message) => {
      busy.current = true;
      setStatus("thinking");
      setError(null);
      const body = {
        direction,
        query,
        focus_id: focusId,
        matches,
        system,
        history: historySnapshot.current,
        message,
      };
      postChat(body, fetchImpl).then(
        (data) => {
          const turn = { role: "assistant", reply: data.reply };
          turnsRef.current = [...turnsRef.current, turn];
          setTurns(turnsRef.current);
          busy.current = false;
          setStatus("idle");
          onTurns?.(turnsRef.current);
        },
        (err) => {
          busy.current = false;
          setError(err);
          setStatus("error");
        },
      );
    },
    [direction, query, focusId, matches, system, fetchImpl, onTurns],
  );

  const send = useCallback(
    (message) => {
      if (busy.current) return;
      const answered = withoutUnansweredTurn(turnsRef.current);
      historySnapshot.current = toHistory(answered);
      lastMessage.current = message;
      turnsRef.current = [...answered, { role: "user", content: message }];
      setTurns(turnsRef.current);
      post(message);
    },
    [post],
  );

  const retry = useCallback(() => {
    if (busy.current || status !== "error") return;
    post(lastMessage.current);
  }, [post, status]);

  const reset = useCallback(() => {
    if (busy.current) return;
    turnsRef.current = [];
    lastMessage.current = null;
    historySnapshot.current = [];
    setTurns([]);
    setStatus("idle");
    setError(null);
    onTurns?.([]);
  }, [onTurns]);

  return { turns, status, error, send, retry, reset };
}

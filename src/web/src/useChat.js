import { useCallback, useRef, useState } from "react";
import { postChat } from "./api.js";
import { toHistory } from "./chatHistory.js";

export function useChat({ direction, query, focusId, matches, system, fetchImpl }) {
  const [turns, setTurns] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState(null);
  const turnsRef = useRef([]);
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
          const turn = {
            role: "assistant",
            reply: data.reply,
            claims: data.claims,
            facts: data.facts,
            suggested_questions: data.suggested_questions,
            off_topic: data.off_topic,
          };
          turnsRef.current = [...turnsRef.current, turn];
          setTurns(turnsRef.current);
          busy.current = false;
          setStatus("idle");
        },
        (err) => {
          busy.current = false;
          setError(err);
          setStatus("error");
        },
      );
    },
    [direction, query, focusId, matches, system, fetchImpl],
  );

  const send = useCallback(
    (message) => {
      if (busy.current) return;
      historySnapshot.current = toHistory(turnsRef.current);
      lastMessage.current = message;
      turnsRef.current = [...turnsRef.current, { role: "user", content: message }];
      setTurns(turnsRef.current);
      post(message);
    },
    [post],
  );

  const retry = useCallback(() => {
    if (busy.current || status !== "error") return;
    post(lastMessage.current);
  }, [post, status]);

  return { turns, status, error, send, retry };
}

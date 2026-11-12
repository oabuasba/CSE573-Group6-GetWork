import { act, renderHook, waitFor } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { useChat } from "./useChat.js";
import agent from "./fixtures/contract_agent.json";

const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const props = (fetchImpl) => ({
  direction: "resume_to_posting",
  query: { kind: "corpus", id: "r-1" },
  focusId: "j-1",
  matches: [{ id: "j-1", rank: 1, score: 0.9 }],
  system: "Full",
  fetchImpl,
});
const bodyOf = (fetchImpl, n) => JSON.parse(fetchImpl.mock.calls[n][1].body);

test("send posts a chat request and appends the reply", async () => {
  const fetchImpl = vi.fn().mockResolvedValue(reply(agent.chat));
  const { result } = renderHook(() => useChat(props(fetchImpl)));
  act(() => result.current.send("Why?"));
  expect(result.current.turns).toEqual([{ role: "user", content: "Why?" }]);
  expect(result.current.status).toBe("thinking");
  await waitFor(() => expect(result.current.status).toBe("idle"));
  expect(bodyOf(fetchImpl, 0)).toEqual({
    direction: "resume_to_posting",
    query: { kind: "corpus", id: "r-1" },
    focus_id: "j-1",
    matches: [{ id: "j-1", rank: 1, score: 0.9 }],
    system: "Full",
    history: [],
    message: "Why?",
  });
  expect(result.current.turns[1]).toEqual({
    role: "assistant",
    reply: agent.chat.reply,
    claims: agent.chat.claims,
    facts: agent.chat.facts,
    suggested_questions: agent.chat.suggested_questions,
    off_topic: false,
  });
});

test("failure keeps the user turn and exposes the error", async () => {
  const fetchImpl = vi.fn().mockResolvedValue(reply(agent.chat_error, 502));
  const { result } = renderHook(() => useChat(props(fetchImpl)));
  act(() => result.current.send("Why?"));
  await waitFor(() => expect(result.current.status).toBe("error"));
  expect(result.current.error.code).toBe("assistant_failed");
  expect(result.current.turns).toEqual([{ role: "user", content: "Why?" }]);
});

test("retry resends without duplicating", async () => {
  const fetchImpl = vi
    .fn()
    .mockResolvedValueOnce(reply(agent.chat_error, 502))
    .mockResolvedValue(reply(agent.chat));
  const { result } = renderHook(() => useChat(props(fetchImpl)));
  act(() => result.current.send("Why?"));
  await waitFor(() => expect(result.current.status).toBe("error"));
  act(() => result.current.retry());
  await waitFor(() => expect(result.current.status).toBe("idle"));
  expect(bodyOf(fetchImpl, 1)).toEqual(bodyOf(fetchImpl, 0));
  expect(result.current.error).toBeNull();
  expect(result.current.turns.map((t) => t.role)).toEqual(["user", "assistant"]);
});

test("history skips an unanswered turn", async () => {
  const fetchImpl = vi
    .fn()
    .mockResolvedValueOnce(reply(agent.chat_error, 502))
    .mockResolvedValue(reply(agent.chat));
  const { result } = renderHook(() => useChat(props(fetchImpl)));
  act(() => result.current.send("first"));
  await waitFor(() => expect(result.current.status).toBe("error"));
  act(() => result.current.send("second"));
  await waitFor(() => expect(result.current.status).toBe("idle"));
  expect(bodyOf(fetchImpl, 1).history).toEqual([]);
  expect(bodyOf(fetchImpl, 1).message).toBe("second");
  act(() => result.current.send("third"));
  await waitFor(() => expect(fetchImpl).toHaveBeenCalledTimes(3));
  expect(bodyOf(fetchImpl, 2).history.map((h) => h.content)).toEqual(["second", agent.chat.reply]);
});

test("a new message after a failure replaces the failed turn", async () => {
  const fetchImpl = vi
    .fn()
    .mockResolvedValueOnce(reply(agent.chat_error, 502))
    .mockResolvedValue(reply(agent.chat));
  const onTurns = vi.fn();
  const { result } = renderHook(() => useChat({ ...props(fetchImpl), onTurns }));
  act(() => result.current.send("first"));
  await waitFor(() => expect(result.current.status).toBe("error"));
  act(() => result.current.send("second"));
  await waitFor(() => expect(result.current.status).toBe("idle"));
  expect(result.current.turns.map((t) => t.content ?? t.reply)).toEqual(["second", agent.chat.reply]);
  expect(onTurns.mock.calls.at(-1)[0].map((t) => t.role)).toEqual(["user", "assistant"]);
});

test("a second send while thinking is ignored", async () => {
  let resolve;
  const fetchImpl = vi.fn(() => new Promise((r) => { resolve = r; }));
  const { result } = renderHook(() => useChat(props(fetchImpl)));
  act(() => result.current.send("one"));
  act(() => result.current.send("two"));
  expect(fetchImpl).toHaveBeenCalledTimes(1);
  expect(result.current.turns).toEqual([{ role: "user", content: "one" }]);
  await act(async () => resolve(reply(agent.chat)));
  await waitFor(() => expect(result.current.status).toBe("idle"));
  act(() => result.current.retry());
  expect(fetchImpl).toHaveBeenCalledTimes(1);
});

const earlier = [
  { role: "user", content: "Why?" },
  { ...agent.chat, role: "assistant" },
];

test("initial turns and history", async () => {
  const fetchImpl = vi.fn().mockResolvedValue(reply(agent.chat));
  const { result } = renderHook(() => useChat({ ...props(fetchImpl), initialTurns: earlier }));
  expect(result.current.turns).toEqual(earlier);
  act(() => result.current.send("And then?"));
  await waitFor(() => expect(result.current.status).toBe("idle"));
  expect(bodyOf(fetchImpl, 0).history).toEqual([
    { role: "user", content: "Why?" },
    { role: "assistant", content: agent.chat.reply },
  ]);
});

test("onTurns and reset", async () => {
  const onTurns = vi.fn();
  let resolve;
  const fetchImpl = vi.fn(() => new Promise((r) => { resolve = r; }));
  const { result } = renderHook(() => useChat({ ...props(fetchImpl), onTurns }));
  act(() => result.current.send("Why?"));
  expect(onTurns).not.toHaveBeenCalled();
  act(() => result.current.reset());
  expect(result.current.turns).toHaveLength(1);
  expect(onTurns).not.toHaveBeenCalled();
  await act(async () => resolve(reply(agent.chat)));
  await waitFor(() => expect(result.current.status).toBe("idle"));
  expect(onTurns).toHaveBeenCalledTimes(1);
  expect(onTurns.mock.calls[0][0]).toHaveLength(2);
  act(() => result.current.reset());
  expect(result.current.turns).toEqual([]);
  expect(result.current.error).toBeNull();
  expect(onTurns).toHaveBeenLastCalledWith([]);
});

test("a failed request does not call onTurns, reset clears the error", async () => {
  const onTurns = vi.fn();
  const fetchImpl = vi.fn().mockResolvedValue(reply(agent.chat_error, 502));
  const { result } = renderHook(() => useChat({ ...props(fetchImpl), onTurns }));
  act(() => result.current.send("Why?"));
  await waitFor(() => expect(result.current.status).toBe("error"));
  expect(onTurns).not.toHaveBeenCalled();
  act(() => result.current.reset());
  expect(result.current.status).toBe("idle");
  expect(result.current.error).toBeNull();
  expect(result.current.turns).toEqual([]);
});

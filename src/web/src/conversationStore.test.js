import { afterEach, expect, test, vi } from "vitest";
import {
  CONVERSATION_PREFIX, clearAllConversations, clearConversation, loadConversation, saveConversation,
} from "./conversationStore.js";
import { loadResults, saveResults } from "./resultsSession.js";

const turns = [
  { role: "user", content: "Why?" },
  { role: "assistant", reply: "Because." },
];

afterEach(() => vi.restoreAllMocks());

test("round trip per match", () => {
  saveConversation("j-1", turns);
  saveConversation("j-2", turns.slice(0, 1));
  expect(loadConversation("j-1")).toEqual(turns);
  expect(loadConversation("j-2")).toEqual(turns.slice(0, 1));
  expect(loadConversation("j-3")).toEqual([]);
  expect(window.sessionStorage.getItem(`${CONVERSATION_PREFIX}j-1`)).not.toBeNull();
  saveConversation("j-1", []);
  expect(window.sessionStorage.getItem(`${CONVERSATION_PREFIX}j-1`)).toBeNull();
  clearConversation("j-2");
  expect(loadConversation("j-2")).toEqual([]);
});

test("old stored turns load as plain replies", () => {
  const old = {
    role: "assistant",
    reply: "Because.",
    claims: [{ text: "c", fact_ids: ["f1"] }],
    facts: [{ id: "f1", text: "t" }],
    suggested_questions: ["More?"],
    off_topic: false,
    note: "Part of the answer was left out.",
  };
  window.sessionStorage.setItem(`${CONVERSATION_PREFIX}j-1`, JSON.stringify([turns[0], old]));
  expect(loadConversation("j-1")).toEqual(turns);
});

test("invalid data and clear all", () => {
  for (const raw of ["nope{", "{}", "[1]", '[{"role":"system"}]', "null"]) {
    window.sessionStorage.setItem(`${CONVERSATION_PREFIX}bad`, raw);
    expect(loadConversation("bad")).toEqual([]);
  }
  for (const bad of [
    { role: "assistant" },
    { role: "assistant", reply: 5 },
    { role: "user", content: 5 },
    { role: "system", reply: "x" },
    null,
  ]) {
    window.sessionStorage.setItem(`${CONVERSATION_PREFIX}bad`, JSON.stringify([turns[0], bad]));
    expect(loadConversation("bad")).toEqual([]);
  }
  saveResults({
    direction: "resume_to_posting",
    response: { query: { kind: "text" }, system: "Full", matches: [] },
    expandedId: null,
  });
  window.sessionStorage.setItem("other", "x");
  for (const id of ["a", "b", "c"]) saveConversation(id, turns);
  clearAllConversations();
  for (const id of ["a", "b", "c"]) expect(loadConversation(id)).toEqual([]);
  expect(loadResults()).not.toBeNull();
  expect(window.sessionStorage.getItem("other")).toBe("x");
});

test("storage errors are swallowed", () => {
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("full");
  });
  expect(() => saveConversation("j-1", turns)).not.toThrow();
  vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
    throw new Error("disabled");
  });
  vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
    throw new Error("disabled");
  });
  expect(loadConversation("j-1")).toEqual([]);
  expect(() => clearConversation("j-1")).not.toThrow();
  expect(() => clearAllConversations()).not.toThrow();
});

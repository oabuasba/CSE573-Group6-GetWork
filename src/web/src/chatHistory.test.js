import { expect, test } from "vitest";
import { MAX_CONTENT, MAX_HISTORY, toHistory } from "./chatHistory.js";

const user = (content) => ({ role: "user", content });
const bot = (reply) => ({ role: "assistant", reply, claims: [], facts: [] });

test("toHistory trims and maps", () => {
  expect(toHistory([user("hi"), bot("hello")])).toEqual([
    { role: "user", content: "hi" },
    { role: "assistant", content: "hello" },
  ]);
  const long = toHistory([user("x".repeat(MAX_CONTENT + 50)), bot("ok")]);
  expect(long[0].content).toHaveLength(MAX_CONTENT);
  const many = Array.from({ length: 15 }, (_, i) => [user(`q${i}`), bot(`a${i}`)]).flat();
  const kept = toHistory(many);
  expect(kept).toHaveLength(MAX_HISTORY);
  expect(kept.at(-1)).toEqual({ role: "assistant", content: "a14" });
  expect(kept[0]).toEqual({ role: "user", content: "q5" });
});

test("toHistory drops unanswered turns", () => {
  expect(toHistory([user("failed"), user("again"), bot("ok")])).toEqual([
    { role: "user", content: "again" },
    { role: "assistant", content: "ok" },
  ]);
  expect(toHistory([user("pending")])).toEqual([]);
  expect(toHistory([user(""), bot("ok"), user("last")])).toEqual([{ role: "assistant", content: "ok" }]);
});

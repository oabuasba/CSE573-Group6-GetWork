import { afterEach, expect, test, vi } from "vitest";
import { RESULTS_KEY, clearResults, loadResults, saveResults, setReturning } from "./resultsSession.js";

const response = {
  query: { kind: "text", title: "t", skills: ["SQL"] },
  system: "Full",
  matches: [{ id: "j-1", rank: 1, score: 0.5 }],
  notes: [],
};
const snapshot = { direction: "resume_to_posting", response, expandedId: "j-1" };

afterEach(() => vi.restoreAllMocks());

test("round trip and returning flag", () => {
  saveResults(snapshot);
  expect(loadResults()).toEqual({ version: 1, ...snapshot, returning: false });
  setReturning(true);
  expect(loadResults().returning).toBe(true);
  expect(loadResults().response).toEqual(response);
  saveResults({ ...snapshot, expandedId: null });
  expect(loadResults()).toMatchObject({ expandedId: null, returning: false });
  clearResults();
  expect(window.sessionStorage.getItem(RESULTS_KEY)).toBeNull();
  setReturning(true);
  expect(window.sessionStorage.getItem(RESULTS_KEY)).toBeNull();
});

test("invalid snapshots load as null", () => {
  const good = { version: 1, ...snapshot, returning: false };
  const cases = [
    null,
    "not json{",
    JSON.stringify({ ...good, version: 2 }),
    JSON.stringify({ ...good, direction: "sideways" }),
    JSON.stringify({ ...good, response: { ...response, matches: "no" } }),
    JSON.stringify({ ...good, response: { ...response, query: undefined } }),
    JSON.stringify({ ...good, response: { ...response, system: undefined } }),
    "123",
  ];
  for (const raw of cases) {
    window.sessionStorage.clear();
    if (raw !== null) window.sessionStorage.setItem(RESULTS_KEY, raw);
    expect(loadResults()).toBeNull();
  }
});

test("storage errors are swallowed", () => {
  saveResults(snapshot);
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new DOMException("full", "QuotaExceededError");
  });
  expect(() => saveResults(snapshot)).not.toThrow();
  expect(() => setReturning(true)).not.toThrow();
  vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
    throw new Error("disabled");
  });
  vi.spyOn(Storage.prototype, "removeItem").mockImplementation(() => {
    throw new Error("disabled");
  });
  expect(loadResults()).toBeNull();
  expect(() => clearResults()).not.toThrow();
});

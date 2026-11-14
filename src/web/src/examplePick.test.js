import { expect, test, vi } from "vitest";
import { pickExamples } from "./examplePick.js";

const seq = (...v) => {
  let i = 0;
  return vi.fn(() => v[Math.min(i++, v.length - 1)]);
};
const make = (n, prefix = "a") => Array.from({ length: n }, (_, i) => ({ id: `${prefix}${i + 1}` }));
const ids = (items) => items.map((item) => item.id);

test("first three are the defaults, in server order", () => {
  const items = make(8);
  const copy = structuredClone(items);
  const result = pickExamples(items, { random: () => 0 });
  expect(ids(result.defaults)).toEqual(["a1", "a2", "a3"]);
  expect(items).toEqual(copy);
  const ten = pickExamples(make(10, "r-"), { random: () => 0 });
  expect(ids(ten.defaults)).toEqual(["r-1", "r-2", "r-3"]);
});

test("samples three distinct non-default items with the injected random", () => {
  const items = make(8);
  const copy = structuredClone(items);
  const low = vi.fn(() => 0);
  expect(ids(pickExamples(items, { random: low }).random)).toEqual(["a4", "a5", "a6"]);
  expect(low).toHaveBeenCalledTimes(3);
  const high = vi.fn(() => 0.99);
  expect(ids(pickExamples(items, { random: high }).random)).toEqual(["a8", "a4", "a5"]);
  expect(high).toHaveBeenCalledTimes(3);
  expect(items).toEqual(copy);
  const ten = make(10, "r-");
  expect(ids(pickExamples(ten, { random: () => 0 }).random)).toEqual(["r-4", "r-5", "r-6"]);
  expect(ids(pickExamples(ten, { random: () => 0.99 }).random)).toEqual(["r-10", "r-4", "r-5"]);
});

test("small pools", () => {
  const empty = pickExamples([], { random: () => 0 });
  expect(empty).toEqual({ defaults: [], random: [] });
  expect(pickExamples(make(2), { random: () => 0 })).toEqual({ defaults: make(2), random: [] });

  const three = vi.fn(() => 0);
  expect(pickExamples(make(3), { random: three }).random).toEqual([]);
  expect(three).not.toHaveBeenCalled();

  expect(ids(pickExamples(make(4), { random: () => 0 }).random)).toEqual(["a4"]);
  expect(ids(pickExamples(make(5), { random: () => 0 }).random)).toEqual(["a4", "a5"]);

  const six = vi.fn(() => 0);
  expect(ids(pickExamples(make(6), { random: six }).random)).toEqual(["a4", "a5", "a6"]);
  expect(six).toHaveBeenCalledTimes(3);
  const reordered = pickExamples(make(6), { random: seq(0.99, 0.99, 0) });
  expect([...ids(reordered.random)].sort()).toEqual(["a4", "a5", "a6"]);
  expect(ids(reordered.random)).toEqual(["a6", "a4", "a5"]);
});

test("duplicate ids and a random source of 1", () => {
  const dupes = [{ id: "a1" }, { id: "a1" }, { id: "a2" }, { id: "a3" }, { id: "a4" }, { id: "a4" }];
  const result = pickExamples(dupes, { random: () => 0 });
  expect(ids(result.defaults)).toEqual(["a1", "a2", "a3"]);
  expect(ids(result.random)).toEqual(["a4"]);

  const one = pickExamples(make(8), { random: () => 1 });
  expect(one.random).toHaveLength(3);
  expect(new Set(ids(one.random)).size).toBe(3);
  for (const item of one.random) expect(["a4", "a5", "a6", "a7", "a8"]).toContain(item.id);
});

test("keeps the chosen example in the random group", () => {
  const random = vi.fn(() => 0);
  const result = pickExamples(make(8), { random, chosenId: "a6" });
  expect(ids(result.random)).toEqual(["a6", "a4", "a5"]);
  expect(random).toHaveBeenCalledTimes(2);
});

test("ignores a chosen id that is a default or unknown", () => {
  const items = make(8);
  const base = ids(pickExamples(items, { random: () => 0 }).random);
  for (const chosenId of ["a2", "zzz", null]) {
    const random = vi.fn(() => 0);
    const result = pickExamples(items, { random, chosenId });
    expect(ids(result.random)).toEqual(base);
    expect(random).toHaveBeenCalledTimes(3);
  }
});

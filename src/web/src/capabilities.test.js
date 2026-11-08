import { describe, expect, test } from "vitest";
import { canRun, parseSkills } from "./capabilities.js";

test("parseSkills", () => {
  expect(parseSkills("Python, SQL;Excel\n  Docker  \n\n,")).toEqual(["Python", "SQL", "Excel", "Docker"]);
  expect(parseSkills("Python, python, PYTHON, SQL")).toEqual(["Python", "SQL"]);
  expect(parseSkills("")).toEqual([]);
  expect(parseSkills(" , ;\n")).toEqual([]);
  const many = Array.from({ length: 150 }, (_, i) => `skill${i}`).join(",");
  const capped = parseSkills(many);
  expect(capped).toHaveLength(100);
  expect(capped[99]).toBe("skill99");
});

describe("canRun matrix", () => {
  const caps = (structure, embed) => ({
    structure: { available: structure, reason: "no structure" },
    embed: { available: embed, reason: "no embed" },
  });

  test.each(["B1+", "B2", "Full"])("corpus is always ok (%s)", (system) => {
    expect(canRun("corpus", system, caps(false, false))).toEqual({ ok: true });
  });

  test.each([
    ["form", "B1+", true, true, true],
    ["form", "B1+", false, false, true],
    ["form", "B2", true, true, true],
    ["form", "B2", true, false, false],
    ["form", "Full", true, false, false],
    ["form", "Full", false, true, true],
    ["text", "B1+", true, false, true],
    ["text", "B1+", false, true, false],
    ["text", "B2", true, true, true],
    ["text", "B2", true, false, false],
    ["text", "Full", false, true, false],
    ["text", "Full", true, true, true],
  ])("%s %s structure=%s embed=%s -> %s", (kind, system, structure, embed, ok) => {
    expect(canRun(kind, system, caps(structure, embed)).ok).toBe(ok);
  });

  test("returns the capability's reason", () => {
    expect(canRun("text", "Full", caps(false, false))).toEqual({ ok: false, reason: "no structure" });
    expect(canRun("text", "Full", caps(true, false))).toEqual({ ok: false, reason: "no embed" });
    expect(canRun("form", "B2", caps(true, false))).toEqual({ ok: false, reason: "no embed" });
  });
});

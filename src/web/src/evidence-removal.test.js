import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

const SRC = import.meta.dirname;
const FORBIDDEN = /ClaimList|Show evidence|Hide evidence|suggested_questions|STARTERS|off_topic/;

function sourceFiles(dir) {
  return readdirSync(dir).flatMap((name) => {
    if (name === "node_modules") return [];
    const path = join(dir, name);
    return statSync(path).isDirectory() ? sourceFiles(path) : [path];
  });
}

test("chat claim UI is gone", () => {
  const files = sourceFiles(SRC);
  expect(files.map((f) => f.split("/").pop())).not.toContain("ClaimList.jsx");
  const sources = files.filter((f) => !/\.test\.(js|jsx)$/.test(f));
  for (const file of sources) {
    expect(readFileSync(file, "utf8"), file).not.toMatch(FORBIDDEN);
  }
});

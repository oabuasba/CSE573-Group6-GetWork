import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "vitest";

const SRC = import.meta.dirname;
const SELF = "chat-removal.test.js";

function sourceFiles(dir) {
  return readdirSync(dir).flatMap((name) => {
    if (name === "node_modules") return [];
    const path = join(dir, name);
    return statSync(path).isDirectory() ? sourceFiles(path) : [path];
  });
}

test("inline chat components are gone", () => {
  const files = sourceFiles(SRC);
  const names = files.map((f) => f.split("/").pop());
  for (const gone of ["ChatOnDemand.jsx", "ChatPanel.jsx", "ChatPanel.test.jsx"]) {
    expect(names).not.toContain(gone);
  }
  for (const file of files.filter((f) => !f.endsWith(SELF))) {
    const text = readFileSync(file, "utf8");
    expect(text, file).not.toMatch(/ChatPanel|ChatOnDemand/);
  }
  const css = readFileSync(join(SRC, "theme", "chat.css"), "utf8");
  expect(css).not.toMatch(/\.chat \{|\.chat-log|\.chat-turn(?!s)/);
});

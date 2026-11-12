import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";

const css = readFileSync(join(import.meta.dirname, "components.css"), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");

const rules = css
  .split("}")
  .filter((chunk) => chunk.includes("{"))
  .map((chunk) => {
    const open = chunk.lastIndexOf("{");
    return { selector: chunk.slice(0, open).trim(), body: chunk.slice(open + 1) };
  });

const rulesFor = (selector) => rules.filter((rule) => rule.selector === selector);
const bodyOf = (selector) => rulesFor(selector).map((rule) => rule.body).join(";");

describe("app shell rules", () => {
  test("app shell rules", () => {
    const shell = bodyOf(".app-shell");
    expect(shell).toMatch(/display:\s*flex/);
    expect(shell).toMatch(/flex-direction:\s*column/);
    expect(shell).toMatch(/min-height:\s*100vh/);
    expect(shell).toMatch(/min-height:\s*100dvh/);
    const main = bodyOf(".app-main");
    expect(main).toMatch(/flex:\s*1 0 auto/);
    expect(main).toMatch(/width:\s*100%/);
    for (const rule of rules.filter((r) => r.selector.includes(".site-footer"))) {
      expect(rule.body).not.toMatch(/position:\s*(fixed|sticky|absolute)/);
    }
  });
});

describe("control row", () => {
  test("control row rules", () => {
    const row = bodyOf(".control-row");
    expect(row).toMatch(/display:\s*flex/);
    expect(row).toMatch(/flex-wrap:\s*wrap/);
    expect(row).toMatch(/gap:/);
    const field = bodyOf(".control-row .field");
    expect(field).toMatch(/flex:\s*1 1 [\d.]+rem/);
    expect(field).toMatch(/min-width:\s*0/);
  });

  test("control row has no fixed widths or media rules", () => {
    const related = rules.filter((rule) => rule.selector.includes(".control-row"));
    expect(related.length).toBeGreaterThan(0);
    for (const rule of related) {
      expect(rule.selector).not.toContain("@media");
      expect(rule.body).not.toMatch(/(^|[;\s])(width|max-width)\s*:/);
      expect(rule.body).not.toMatch(/\d+px/);
    }
  });
});

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";

const SRC_DIR = join(import.meta.dirname, "..");

function cssFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return cssFiles(path);
    return entry.name.endsWith(".css") ? [path] : [];
  });
}

const sources = cssFiles(SRC_DIR).map((path) => readFileSync(path, "utf8"));
const allCss = sources.join("\n").replace(/\/\*[\s\S]*?\*\//g, "");

// Simple splitting: every chunk before "}" is "[@media ... {] selector { declarations".
const rules = allCss
  .split("}")
  .filter((chunk) => chunk.includes("{"))
  .map((chunk) => {
    const open = chunk.lastIndexOf("{");
    const head = chunk.slice(0, open);
    const media = head.match(/@media[^{]*\{/)?.[0] ?? "";
    return {
      selector: head.replace(media, "").trim(),
      media,
      body: chunk.slice(open + 1),
    };
  });

const declares = (rule, property) => new RegExp(`(^|[;\\s])${property}\\s*:`).test(rule.body);
const withoutNot = (selector) => selector.replace(/:not\([^)]*\)/g, "");
const isLinkSelector = (selector) =>
  selector.split(",").every((part) => {
    const last = withoutNot(part).trim().split(/\s+/).pop().replace(/(:[\w-]+|\[[^\]]*\])+$/, "");
    return last === "a" || last === "a.skip-link" || last === ".link";
  });

describe("style rules", () => {
  test("rules were found", () => {
    expect(sources.length).toBeGreaterThan(1);
    expect(rules.length).toBeGreaterThan(20);
  });

  test("brand typography rules", () => {
    expect(allCss).not.toMatch(/font-style\s*:\s*italic/);
    expect(allCss).not.toMatch(/text-transform\s*:\s*uppercase/);
    for (const match of allCss.matchAll(/font-family\s*:\s*([^;}]+)/g)) {
      expect(match[1].trim()).toBe("var(--font-stack)");
    }
    const underlined = rules.filter((r) => /text-decoration(-line)?\s*:[^;]*underline/.test(r.body));
    for (const rule of underlined) {
      expect(isLinkSelector(rule.selector), `underline on ${rule.selector}`).toBe(true);
    }
    for (const match of allCss.matchAll(/font(?:-size)?\s*:\s*([^;}]+)/g)) {
      expect(match[1], "font size must not use px").not.toMatch(/\d\s*px/);
    }
    const measure = rules.find((r) => r.selector === ".measure");
    expect(measure?.body).toMatch(/max-width\s*:\s*var\(--measure\)/);
  });

  test("button rules", () => {
    const buttonRules = rules.filter((r) => /\.btn|button|role=/.test(withoutNot(r.selector)));
    expect(buttonRules.length).toBeGreaterThan(0);
    for (const rule of buttonRules) {
      expect(declares(rule, "box-shadow"), `box-shadow in ${rule.selector}`).toBe(false);
    }
    const hoverRules = buttonRules.filter((r) => /:hover/.test(r.selector));
    const grows = hoverRules.filter((r) => r.body.includes("transform: scale(var(--scale-hover))"));
    expect(grows.length).toBeGreaterThan(0);
    for (const rule of hoverRules) {
      expect(declares(rule, "color"), `color in ${rule.selector}`).toBe(false);
      const decoration = declares(rule, "text-decoration(-line)?");
      expect(decoration, `underline in ${rule.selector}`).toBe(false);
    }
    const base = rules.find((r) => r.selector === ".btn");
    expect(base.body).toMatch(/transition\s*:\s*transform/);
  });

  test("focus and motion rules", () => {
    const focus = rules.find((r) => r.selector === ":focus-visible");
    expect(focus.body).toMatch(/outline\s*:\s*var\(--focus-width\) solid var\(--focus-ring\)/);
    const reduced = rules.filter((r) => r.media.includes("prefers-reduced-motion: reduce"));
    expect(reduced.some((r) => /transform\s*:\s*none/.test(r.body))).toBe(true);
    expect(reduced.some((r) => /transition\s*:\s*none/.test(r.body))).toBe(true);
  });

  test("no external urls in css", () => {
    expect(allCss).not.toMatch(/https?:\/\//);
  });
});

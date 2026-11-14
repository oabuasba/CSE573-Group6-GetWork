import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { NON_TEXT_PAIRS, TEXT_PAIRS } from "./pairs.js";

const read = (name) => readFileSync(join(import.meta.dirname, name), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");
const tokensCss = read("tokens.css");

function parseRules(css) {
  return css
    .split("}")
    .filter((chunk) => chunk.includes("{"))
    .map((chunk) => {
      const open = chunk.lastIndexOf("{");
      return { selector: chunk.slice(0, open).trim().replace(/\s+/g, " "), body: chunk.slice(open + 1) };
    });
}

const homeCss = read("home.css");
const homeRules = parseRules(homeCss);
const bodyOf = (rules, selector) =>
  rules.filter((rule) => rule.selector === selector).map((rule) => rule.body).join(";");
const hasDeclaration = (body, property) => new RegExp(`(^|[;\\s])${property}\\s*:`).test(body);

function tokenValue(name) {
  let value = tokensCss.match(new RegExp(`${name}\\s*:\\s*([^;]+);`))?.[1].trim();
  while (value?.startsWith("var(")) {
    value = tokensCss.match(new RegExp(`${value.slice(4, -1).trim()}\\s*:\\s*([^;]+);`))?.[1].trim();
  }
  return value;
}

describe("hero tokens", () => {
  test("hero tokens are defined and paired", () => {
    expect(tokenValue("--hero-bg")).toBe("#8C1D40");
    expect(tokenValue("--hero-fg")).toBe("#FFFFFF");
    expect(tokenValue("--text-hero")).toBe("2.5rem");
    expect(tokenValue("--text-lead")).toBe("1.25rem");
    const text = TEXT_PAIRS.map((pair) => pair.join("|"));
    expect(text).toContain("--hero-fg|--hero-bg|4.5");
    const nonText = NON_TEXT_PAIRS.map((pair) => pair.join("|"));
    for (const foreground of ["--focus-ring-on-dark", "--btn-cta-bg", "--btn-secondary-bg"]) {
      expect(nonText).toContain(`${foreground}|--hero-bg|3`);
    }
  });
});

describe("home.css", () => {
  test("home.css declares the required rules", () => {
    const expectations = {
      ".home-hero": [/background:\s*var\(--hero-bg\)/, /color:\s*var\(--hero-fg\)/, /border-radius:\s*var\(--radius-card\)/, /padding:\s*var\(--space-/],
      ".home-hero h1": [/font-size:\s*var\(--text-hero\)/, /max-width:\s*var\(--measure\)/],
      ".home-hero :focus-visible": [/outline-color:\s*var\(--focus-ring-on-dark\)/],
      ".home-lead": [/font-size:\s*var\(--text-lead\)/, /max-width:\s*var\(--measure\)/],
      ".home-actions": [/display:\s*flex/, /flex-wrap:\s*wrap/, /gap:\s*var\(--space-2\)/, /margin:\s*var\(--space-/],
      ".home-hint": [/font-size:\s*var\(--text-small\)/, /margin:\s*0/],
      ".home-section": [/margin-top:\s*var\(--space-8\)/],
      ".home-steps, .home-features": [/list-style:\s*none/, /margin:\s*0/, /padding:\s*0/, /display:\s*grid/, /gap:\s*var\(--space-3\)/, /grid-template-columns:\s*repeat\(auto-fit,\s*minmax\(14rem,\s*1fr\)\)/],
      ".home-steps": [/counter-reset:\s*home-step/],
      ".home-steps > li, .home-features > li": [/margin-bottom:\s*0/],
      ".home-step::before": [
        /counter-increment:\s*home-step/,
        /content:\s*counter\(home-step\);/,
        /content:\s*counter\(home-step\)\s*\/\s*""/,
        /display:\s*inline-flex/,
        /width:\s*2rem/,
        /height:\s*2rem/,
        /border-radius:\s*50%/,
        /background:\s*var\(--btn-primary-bg\)/,
        /color:\s*var\(--btn-primary-fg\)/,
        /font-weight:\s*bold/,
        /margin-bottom:\s*var\(--space-2\)/,
      ],
      ".home-cta": [/margin-bottom:\s*0/],
    };
    for (const [selector, patterns] of Object.entries(expectations)) {
      const body = bodyOf(homeRules, selector);
      expect(body, `rule ${selector}`).not.toBe("");
      for (const pattern of patterns) expect(body, `${selector} ${pattern}`).toMatch(pattern);
    }
  });

  test("anchor buttons and the wordmark are not underlined", () => {
    const rules = parseRules(read("components.css"));
    expect(bodyOf(rules, "a.btn, a.wordmark")).toMatch(/text-decoration:\s*none/);
    for (const rule of rules.filter((r) => /\.btn[^{]*:hover/.test(r.selector))) {
      expect(hasDeclaration(rule.body, "text-decoration(-line)?"), rule.selector).toBe(false);
    }
  });

  test("home layout wraps without media queries or fixed widths", () => {
    expect(homeCss).not.toContain("@media");
    expect(homeCss).not.toMatch(/\d\s*px/);
    expect(homeCss).not.toMatch(/url\(|https?:/);
    const sized = homeRules.filter((rule) => hasDeclaration(rule.body, "(max-)?width"));
    expect(sized.map((rule) => rule.selector).sort()).toEqual(
      [".home-hero h1", ".home-lead", ".home-step::before"].sort(),
    );
    for (const selector of [".home-hero", ".home-steps", ".home-features", ".home-actions"]) {
      expect(hasDeclaration(bodyOf(homeRules, selector), "(max-)?width"), selector).toBe(false);
    }
    const columns = homeCss.match(/grid-template-columns:\s*repeat\(auto-fit,\s*minmax\((\d+(?:\.\d+)?)rem,\s*1fr\)\)/);
    expect(Number(columns?.[1])).toBeGreaterThanOrEqual(10);
    expect(Number(columns?.[1])).toBeLessThanOrEqual(16);
  });
});

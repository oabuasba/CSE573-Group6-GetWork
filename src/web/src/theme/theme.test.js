import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { FORBIDDEN_PAIRS, NON_TEXT_PAIRS, TEXT_PAIRS } from "./pairs.js";

const css = readFileSync(join(import.meta.dirname, "tokens.css"), "utf8");
const tokens = Object.fromEntries(
  [...css.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)].map(([, name, value]) => [name, value.trim()]),
);

function resolve(name) {
  let value = tokens[name];
  for (let hops = 0; value?.startsWith("var("); hops += 1) {
    expect(hops, `var() cycle at ${name}`).toBeLessThan(10);
    value = tokens[value.slice(4, -1).trim()];
  }
  return value;
}

function luminance(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => {
    const c = parseInt(hex.slice(i, i + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(foreground, background) {
  const [a, b] = [luminance(resolve(foreground)), luminance(resolve(background))];
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

const PALETTE = {
  "--color-maroon": "#8C1D40",
  "--color-gold": "#FFC627",
  "--color-black": "#000000",
  "--color-white": "#FFFFFF",
  "--color-green": "#78BE20",
  "--color-blue": "#00A3E0",
  "--color-orange": "#FF7F32",
  "--color-copper": "#AF674B",
  "--color-turquoise": "#4AB7C4",
  "--color-pink": "#E74973",
  "--color-gray-1": "#191919",
  "--color-gray-2": "#484848",
  "--color-gray-3": "#747474",
  "--color-gray-4": "#BFBFBF",
  "--color-gray-5": "#D0D0D0",
  "--color-gray-6": "#E8E8E8",
  "--color-gray-7": "#FAFAFA",
};

describe("theme tokens", () => {
  test("palette matches the brand guide values", () => {
    for (const [name, hex] of Object.entries(PALETTE)) {
      expect(tokens[name], name).toBe(hex);
    }
  });

  test("font stack and spacing rules", () => {
    expect(tokens["--font-stack"]).toBe(
      'Arial, Helvetica, "Nimbus Sans L", "Liberation Sans", FreeSans, sans-serif',
    );
    expect(tokens["--measure"]).toBe("43.75rem");
    expect(tokens["--text-base"]).toBe("1rem");
    const spaces = Object.entries(tokens).filter(([name]) => /^--space-\d+$/.test(name));
    expect(spaces.length).toBeGreaterThan(0);
    for (const [name, value] of spaces) {
      expect(parseFloat(value) % 0.5, name).toBe(0);
      expect(value.endsWith("rem"), name).toBe(true);
    }
  });

  test("all declared pairs meet contrast thresholds", () => {
    for (const [foreground, background, minimum] of [...TEXT_PAIRS, ...NON_TEXT_PAIRS]) {
      const ratio = contrast(foreground, background);
      expect(ratio, `${foreground} on ${background}`).toBeGreaterThanOrEqual(minimum);
    }
    expect(TEXT_PAIRS.every(([, , minimum]) => minimum === 4.5)).toBe(true);
    expect(NON_TEXT_PAIRS.every(([, , minimum]) => minimum === 3)).toBe(true);
  });

  test("forbidden pairs are absent", () => {
    const declared = [...TEXT_PAIRS, ...NON_TEXT_PAIRS].map(([f, b]) => [resolve(f), resolve(b)]);
    for (const [foreground, background] of FORBIDDEN_PAIRS) {
      const forbidden = [resolve(foreground), resolve(background)];
      expect(declared).not.toContainEqual(forbidden);
    }
    const mutedOnBg = [...TEXT_PAIRS, ...NON_TEXT_PAIRS].some(
      ([f, b]) => f === "--text-muted" && b === "--bg",
    );
    expect(mutedOnBg).toBe(false);
  });
});

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";

const raw = readFileSync(join(import.meta.dirname, "chat-split.css"), "utf8");
const css = raw.replace(/\/\*[\s\S]*?\*\//g, "");
const mediaAt = css.indexOf("@media");
const baseCss = css.slice(0, mediaAt);
const mediaCss = css.slice(mediaAt);

function bodyIn(source, selector) {
  return source
    .split("}")
    .filter((chunk) => chunk.includes("{"))
    .map((chunk) => ({
      selector: chunk.slice(0, chunk.lastIndexOf("{")).replace(/@media[^{]*\{/, "").trim(),
      body: chunk.slice(chunk.lastIndexOf("{") + 1),
    }))
    .filter((rule) => rule.selector === selector)
    .map((rule) => rule.body)
    .join(";");
}

const bodyOf = (selector, inMedia = false) => bodyIn(inMedia ? mediaCss : baseCss, selector);

describe("chat split layout", () => {
  test("wide screens use a two column grid", () => {
    const split = bodyOf(".chat-split");
    expect(split).toMatch(/display:\s*grid/);
    expect(split).toMatch(/grid-template-columns:\s*minmax\(0,\s*1fr\)\s+minmax\(18rem,\s*26rem\)\s*;/);
    expect(split).toMatch(/grid-template-rows:\s*minmax\(0,\s*1fr\)/);
    const fallback = split.indexOf("height: clamp(28rem, calc(100vh - 22rem), 52rem)");
    const modern = split.indexOf("height: clamp(28rem, calc(100dvh - 22rem), 52rem)");
    expect(fallback).toBeGreaterThanOrEqual(0);
    expect(modern).toBeGreaterThan(fallback);
  });

  test("solo mode is one centered column", () => {
    expect(bodyOf(".chat-split-solo")).toMatch(/grid-template-columns:\s*minmax\(0,\s*1fr\)\s*;?\s*$/);
    const column = bodyOf(".chat-split-solo .chat-column");
    expect(column).toMatch(/max-width:\s*52rem/);
    expect(column).toMatch(/justify-self:\s*center/);
  });

  test("hidden panel is not displayed", () => {
    expect(bodyOf(".chat-suggestion-panel")).not.toMatch(/(^|[;\s])display\s*:/);
    expect(bodyOf(".chat-suggestion-panel[hidden]")).toMatch(/display:\s*none/);
  });

  test("narrow screens stack the panes", () => {
    const queries = raw.match(/@media[^{]*\{/g) ?? [];
    expect(queries).toHaveLength(1);
    expect(queries[0]).toMatch(/\(max-width:\s*48rem\)/);
    expect(queries[0]).not.toMatch(/px/);
    const split = bodyOf(".chat-split", true);
    expect(split).toMatch(/grid-template-columns:\s*minmax\(0,\s*1fr\)\s*;/);
    expect(split).toMatch(/height:\s*auto/);
    expect(bodyOf(".chat-column", true)).toMatch(/height:\s*clamp\(24rem,\s*70dvh,\s*40rem\)/);
  });

  test("the thread scrolls and the composer stays at the bottom of the chat column", () => {
    const column = bodyOf(".chat-column");
    expect(column).toMatch(/display:\s*flex/);
    expect(column).toMatch(/flex-direction:\s*column/);
    expect(column).toMatch(/min-height:\s*0/);
    const scroll = bodyOf(".chat-scroll");
    expect(scroll).toMatch(/flex:\s*1 1 0\s*;/);
    expect(scroll).toMatch(/min-height:\s*0/);
    expect(scroll).toMatch(/overflow-y:\s*auto/);
    expect(bodyOf(".chat-suggestion-panel")).toMatch(/overflow-y:\s*auto/);
  });

  test("chat-split.css has no raw colors, sticky or fixed positioning", () => {
    expect(css).not.toMatch(/#[0-9a-f]{3,8}\b/i);
    expect(css).not.toMatch(/(rgb|hsl)a?\(/);
    expect(css).not.toMatch(/position:\s*(sticky|fixed)/);
  });
});

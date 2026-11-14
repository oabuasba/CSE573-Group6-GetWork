import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";

const css = readFileSync(join(import.meta.dirname, "chat.css"), "utf8").replace(/\/\*[\s\S]*?\*\//g, "");

const rules = css
  .split("}")
  .filter((chunk) => chunk.includes("{"))
  .map((chunk) => {
    const open = chunk.lastIndexOf("{");
    return { selector: chunk.slice(0, open).trim(), body: chunk.slice(open + 1) };
  });

const rule = (selector) => {
  const found = rules.find((r) => r.selector === selector);
  expect(found, `rule ${selector}`).toBeDefined();
  return found.body;
};
const declares = (body, property) => new RegExp(`(^|[;\\s])${property}\\s*:`).test(body);

describe("chat.css composer", () => {
  test("composer textarea uses the UI font", () => {
    const body = rule(".chat-composer-box textarea");
    expect(body).toMatch(/font-family:\s*var\(--font-stack\)/);
    expect(body).toMatch(/font-size:\s*var\(--text-base\)/);
    expect(body).not.toMatch(/monospace/);
  });

  test("composer is not sticky and has no top border", () => {
    const body = rule(".chat-composer");
    expect(declares(body, "position")).toBe(false);
    expect(declares(body, "border-top")).toBe(false);
    expect(css).not.toMatch(/position\s*:\s*sticky/);
  });

  test("composer box shape and focus ring", () => {
    const body = rule(".chat-composer-box");
    expect(body).toMatch(/border:\s*1px solid var\(--border-strong\)/);
    expect(body).toMatch(/border-radius:\s*calc\(var\(--radius-card\) \* 3\)/);
    const focus = rule(".chat-composer-box:focus-within");
    expect(focus).toMatch(/outline:\s*var\(--focus-width\) solid var\(--focus-ring\)/);
    expect(rule(".chat-composer-box textarea:focus-visible")).toMatch(/outline:\s*none/);
  });

  test("send button is a 2.5rem circle", () => {
    const body = rule(".chat-composer .chat-send");
    expect(body).toMatch(/width:\s*2\.5rem/);
    expect(body).toMatch(/height:\s*2\.5rem/);
    expect(body).toMatch(/border-radius:\s*50%/);
  });

  test("counter is small and muted", () => {
    const body = rule(".chat-counter");
    expect(body).toMatch(/color:\s*var\(--text-muted\)/);
    expect(body).toMatch(/font-size:\s*0\.75rem/);
    expect(rule(".chat-composer-box textarea::placeholder")).toMatch(/color:\s*var\(--text-muted\)/);
  });

  test("textarea is capped at six lines", () => {
    const body = rule(".chat-composer-box textarea");
    expect(body).toMatch(/max-height:\s*9rem/);
    expect(body).toMatch(/overflow-y:\s*auto/);
  });

  test("bubbles wrap long words", () => {
    expect(rule(".chat-bubble")).toMatch(/overflow-wrap:\s*anywhere/);
  });

  test("disclaimer is small and muted", () => {
    const body = rule(".chat-disclaimer");
    expect(body).toMatch(/font-size:\s*0\.75rem/);
    expect(body).toMatch(/color:\s*var\(--text-muted\)/);
  });

  test("assistant replies are plain text on the surface", () => {
    expect(css).not.toMatch(/\.chat-suggestions|\.chat-note/);
    const reply = rule(".chat-reply");
    expect(reply).toMatch(/white-space:\s*pre-wrap/);
    expect(reply).toMatch(/overflow-wrap:\s*anywhere/);
    const assistant = rule(".chat-bubble-assistant");
    expect(assistant).toMatch(/border:\s*0/);
    expect(assistant).toMatch(/background:\s*transparent/);
  });

  test("chat.css has no raw colors", () => {
    expect(css).not.toMatch(/#[0-9a-f]{3,8}\b/i);
    expect(css).not.toMatch(/rgb\(|hsl\(/);
  });
});

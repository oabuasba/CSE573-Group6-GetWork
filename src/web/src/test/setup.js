import "@testing-library/jest-dom/vitest";
import * as axeMatchers from "vitest-axe/matchers";
import { afterEach, expect, vi } from "vitest";

expect.extend(axeMatchers);

// The dev-mock and vite-config tests run in the node environment, where there is no window.
if (typeof window !== "undefined") window.scrollTo = vi.fn();

afterEach(() => {
  globalThis.sessionStorage?.clear();
});

import "@testing-library/jest-dom/vitest";
import * as axeMatchers from "vitest-axe/matchers";
import { afterEach, expect } from "vitest";

expect.extend(axeMatchers);

afterEach(() => {
  globalThis.sessionStorage?.clear();
});

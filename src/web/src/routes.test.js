import { expect, test } from "vitest";
import { DIRECTION_BY_ROUTE, ROUTES, hrefFor, parseRoute } from "./routes.js";

test("parseRoute and hrefFor", () => {
  expect(parseRoute("#/candidates")).toBe("candidates");
  expect(parseRoute("#/about")).toBe("about");
  expect(parseRoute("#/jobs")).toBe("jobs");
  expect(parseRoute("")).toBe("jobs");
  expect(parseRoute(undefined)).toBe("jobs");
  expect(parseRoute("#/nope")).toBe("jobs");
  expect(parseRoute("#/candidates/extra")).toBe("jobs");
  for (const route of ROUTES) expect(parseRoute(hrefFor(route))).toBe(route);
  expect(hrefFor("jobs")).toBe("#/jobs");
  expect(DIRECTION_BY_ROUTE).toEqual({ jobs: "resume_to_posting", candidates: "posting_to_resume" });
});

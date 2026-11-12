import { expect, test } from "vitest";
import {
  DIRECTION_BY_ROUTE, ROUTES, ROUTE_BY_DIRECTION, chatHref, chatIdFromHash, hrefFor, parseRoute,
} from "./routes.js";

test("parseRoute keeps old behavior", () => {
  expect(parseRoute("#/candidates")).toBe("candidates");
  expect(parseRoute("#/about")).toBe("about");
  expect(parseRoute("#/jobs")).toBe("jobs");
  expect(parseRoute("")).toBe("jobs");
  expect(parseRoute(undefined)).toBe("jobs");
  expect(parseRoute("#/nope")).toBe("jobs");
  for (const route of ROUTES) expect(parseRoute(hrefFor(route))).toBe(route);
  expect(hrefFor("jobs")).toBe("#/jobs");
  expect(DIRECTION_BY_ROUTE).toEqual({ jobs: "resume_to_posting", candidates: "posting_to_resume" });
  expect(ROUTE_BY_DIRECTION).toEqual({ resume_to_posting: "jobs", posting_to_resume: "candidates" });
});

test("chat route and id helpers", () => {
  expect(parseRoute("#/chat")).toBe("chat");
  expect(parseRoute("#/chat/j-1")).toBe("chat");
  expect(parseRoute("#/candidates/extra")).toBe("candidates");
  expect(chatIdFromHash("#/chat/j-1")).toBe("j-1");
  for (const hash of ["#/chat", "#/chat/", "#/jobs", "#/chat/%E0%A4%A", "", undefined]) {
    expect(chatIdFromHash(hash)).toBeNull();
  }
});

test("chatHref round trip", () => {
  expect(chatHref("j-1")).toBe("#/chat/j-1");
  for (const id of ["j-1", "a b", "x/y", "p#q?r", "é"]) {
    expect(chatIdFromHash(chatHref(id))).toBe(id);
    expect(parseRoute(chatHref(id))).toBe("chat");
  }
});

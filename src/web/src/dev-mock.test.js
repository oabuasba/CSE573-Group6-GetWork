// @vitest-environment node
import { expect, test } from "vitest";
import { mockResponse } from "../dev-mock.js";
import contract from "./fixtures/contract.json";

test("serves fixture sections", () => {
  expect(mockResponse("GET", "/api/health")).toEqual({ status: 200, body: contract.health });
  const examples = mockResponse("GET", "/api/examples?side=resume&limit=6");
  expect(examples).toEqual({ status: 200, body: contract.examples });
  expect(mockResponse("POST", "/api/match")).toEqual({ status: 200, body: contract.match });
  expect(mockResponse("POST", "/api/explain")).toEqual({ status: 200, body: contract.explain });
});

test("unknown path gives a 404 error envelope", () => {
  const { status, body } = mockResponse("GET", "/api/nope");
  expect(status).toBe(404);
  expect(Object.keys(body)).toEqual(Object.keys(contract.error));
  expect(Object.keys(body.error)).toEqual(Object.keys(contract.error.error));
  expect(body.error.code).toBe("not_found");
});

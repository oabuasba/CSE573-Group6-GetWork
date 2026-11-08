import { describe, expect, test, vi } from "vitest";
import {
  ApiError,
  BAD_RESPONSE_CODE,
  NETWORK_ERROR_CODE,
  describeError,
  getExamples,
  getHealth,
  postExplain,
  postMatch,
} from "./api.js";
import contract from "./fixtures/contract.json";

const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const fakeFetch = (data, status) => vi.fn().mockResolvedValue(reply(data, status));

async function caught(promise) {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("expected a rejection");
}

test("calls the documented endpoints", async () => {
  const f = fakeFetch(contract.health);
  expect(await getHealth(f)).toEqual(contract.health);
  expect(f).toHaveBeenCalledWith("/api/health", expect.objectContaining({ method: "GET" }));

  const g = fakeFetch(contract.examples);
  expect(await getExamples("resume", 6, g)).toEqual(contract.examples);
  expect(g.mock.calls[0][0]).toBe("/api/examples?side=resume&limit=6");

  const body = { direction: "resume_to_posting", k: 10, system: "Full", query: { kind: "corpus", id: "r-1" } };
  const m = fakeFetch(contract.match);
  expect(await postMatch(body, m)).toEqual(contract.match);
  expect(m.mock.calls[0][0]).toBe("/api/match");
  expect(m.mock.calls[0][1]).toMatchObject({ method: "POST", headers: { "Content-Type": "application/json" } });
  expect(JSON.parse(m.mock.calls[0][1].body)).toEqual(body);

  const explainBody = { direction: "resume_to_posting", query: contract.match.query, match_id: "j-1" };
  const e = fakeFetch(contract.explain);
  expect(await postExplain(explainBody, e)).toEqual(contract.explain);
  expect(e.mock.calls[0][0]).toBe("/api/explain");
  expect(JSON.parse(e.mock.calls[0][1].body)).toEqual(explainBody);
});

test("maps error envelopes", async () => {
  const error = await caught(getHealth(fakeFetch(contract.error, 409)));
  expect(error).toBeInstanceOf(ApiError);
  expect(error).toMatchObject({ status: 409, ...contract.error.error });
});

test("maps network and bad responses", async () => {
  const down = await caught(getHealth(vi.fn().mockRejectedValue(new TypeError("fail"))));
  expect(down).toMatchObject({ status: 0, code: NETWORK_ERROR_CODE });

  const notJson = { ok: true, status: 200, json: async () => { throw new SyntaxError("x"); } };
  const bad = await caught(getHealth(vi.fn().mockResolvedValue(notJson)));
  expect(bad.code).toBe(BAD_RESPONSE_CODE);

  const odd = await caught(getHealth(fakeFetch({ nope: 1 }, 500)));
  expect(odd.code).toBe(BAD_RESPONSE_CODE);
});

describe("describeError titles", () => {
  const titles = {
    network_unreachable: "Cannot reach the GetWork server",
    bad_response: "The server sent an unexpected reply",
    empty_query: "Add some text or skills first",
    text_too_long: "That text is too long",
    side_mismatch: "That document is on the wrong side",
    not_found: "We could not find that document",
    embedding_mismatch: "The embedding model does not match the stored data",
    structuring_failed: "The language model could not read that document",
    embedding_failed: "The embedding step failed",
    capability_unavailable: "That option is not available on this server",
    invalid_request: "Please check the form",
    internal: "Something went wrong",
  };
  test.each(Object.entries(titles))("%s", (code, title) => {
    const err = new ApiError({ status: 400, code, message: "m", hint: "h" });
    expect(describeError(err)).toEqual({ title, body: "m", hint: "h" });
  });
  test("unknown code and non-ApiError use the generic title", () => {
    const unknown = new ApiError({ status: 400, code: "weird", message: "m" });
    expect(describeError(unknown).title).toBe("Something went wrong");
    expect(describeError(new Error("boom")).title).toBe("Something went wrong");
  });
});

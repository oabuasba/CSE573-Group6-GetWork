export const NETWORK_ERROR_CODE = "network_unreachable";
export const BAD_RESPONSE_CODE = "bad_response";
export const SYSTEM_LABELS = {
  Full: "GetWork (skill clusters)",
  B2: "Embeddings only",
  "B1+": "Keywords only",
};

const GENERIC_TITLE = "Something went wrong";
const ERROR_TITLES = {
  [NETWORK_ERROR_CODE]: "Cannot reach the GetWork server",
  [BAD_RESPONSE_CODE]: "The server sent an unexpected reply",
  empty_query: "Add some text or skills first",
  text_too_long: "That text is too long",
  side_mismatch: "That document is on the wrong side",
  not_found: "We could not find that document",
  embedding_mismatch: "The embedding model does not match the stored data",
  structuring_failed: "The language model could not read that document",
  embedding_failed: "The embedding step failed",
  capability_unavailable: "That option is not available on this server",
  invalid_request: "Please check the form",
};

export class ApiError extends Error {
  constructor({ status, code, message, hint = null, field = null }) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.hint = hint;
    this.field = field;
  }
}

function badResponse(status) {
  return new ApiError({
    status,
    code: BAD_RESPONSE_CODE,
    message: "The server sent a reply the page could not read.",
  });
}

async function request(method, path, body, fetchImpl) {
  let response;
  try {
    response = await fetchImpl(path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError({
      status: 0,
      code: NETWORK_ERROR_CODE,
      message: "Cannot reach the GetWork server.",
    });
  }
  let data;
  try {
    data = await response.json();
  } catch {
    throw badResponse(response.status);
  }
  if (response.ok) return data;
  if (data && data.error && typeof data.error.code === "string") {
    const { code, message, hint, field } = data.error;
    throw new ApiError({ status: response.status, code, message, hint, field });
  }
  throw badResponse(response.status);
}

export const getHealth = (fetchImpl = fetch) => request("GET", "/api/health", undefined, fetchImpl);

export const getExamples = (side, limit = 6, fetchImpl = fetch) =>
  request("GET", `/api/examples?side=${encodeURIComponent(side)}&limit=${limit}`, undefined, fetchImpl);

export const postMatch = (body, fetchImpl = fetch) => request("POST", "/api/match", body, fetchImpl);

export const postExplain = (body, fetchImpl = fetch) => request("POST", "/api/explain", body, fetchImpl);

export function describeError(err) {
  if (!(err instanceof ApiError)) {
    return { title: GENERIC_TITLE, body: err?.message ?? "", hint: null };
  }
  return { title: ERROR_TITLES[err.code] ?? GENERIC_TITLE, body: err.message, hint: err.hint };
}

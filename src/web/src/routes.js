export const ROUTES = ["home", "jobs", "candidates", "about", "chat"];
export const DIRECTION_BY_ROUTE = { jobs: "resume_to_posting", candidates: "posting_to_resume" };
export const ROUTE_BY_DIRECTION = { resume_to_posting: "jobs", posting_to_resume: "candidates" };

// First path segment after "#/", if it is a known route, else "home" ("#/chat/j-1" -> "chat").
export function parseRoute(hash) {
  const name = String(hash ?? "").replace(/^#\/?/, "").split("/")[0];
  return ROUTES.includes(name) ? name : "home";
}

// "#/chat/<encoded id>" -> decoded id; null for other routes, an empty id or a malformed percent sequence.
export function chatIdFromHash(hash) {
  const match = /^#\/?chat\/(.+)$/.exec(String(hash ?? ""));
  if (!match) return null;
  try {
    return decodeURIComponent(match[1]);
  } catch {
    return null;
  }
}

export function chatHref(matchId) {
  return `#/chat/${encodeURIComponent(matchId)}`;
}

export function hrefFor(route) {
  return `#/${route}`;
}

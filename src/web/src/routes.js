export const ROUTES = ["jobs", "candidates", "about"];
export const DIRECTION_BY_ROUTE = { jobs: "resume_to_posting", candidates: "posting_to_resume" };

export function parseRoute(hash) {
  const name = String(hash ?? "").replace(/^#\/?/, "");
  return ROUTES.includes(name) ? name : "jobs";
}

export function hrefFor(route) {
  return `#/${route}`;
}

export const RESULTS_KEY = "getwork.results.v1";
const DIRECTIONS = ["resume_to_posting", "posting_to_resume"];

function isValidSnapshot(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    value.version === 1 &&
    DIRECTIONS.includes(value.direction) &&
    Array.isArray(value.response?.matches) &&
    Boolean(value.response.query) &&
    Boolean(value.response.system)
  );
}

function writeSnapshot(snapshot) {
  try {
    window.sessionStorage.setItem(RESULTS_KEY, JSON.stringify(snapshot));
  } catch {
    // Storage full or disabled: the page works without the snapshot.
  }
}

export function saveResults({ direction, response, expandedId }) {
  writeSnapshot({ version: 1, direction, response, expandedId: expandedId ?? null, returning: false });
}

export function loadResults() {
  try {
    const raw = window.sessionStorage.getItem(RESULTS_KEY);
    if (raw === null) return null;
    const snapshot = JSON.parse(raw);
    return isValidSnapshot(snapshot) ? snapshot : null;
  } catch {
    return null;
  }
}

export function setReturning(value) {
  const snapshot = loadResults();
  if (!snapshot) return;
  writeSnapshot({ ...snapshot, returning: Boolean(value) });
}

export function clearResults() {
  try {
    window.sessionStorage.removeItem(RESULTS_KEY);
  } catch {
    // Storage disabled: nothing to clear.
  }
}

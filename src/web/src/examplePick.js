export const DEFAULT_COUNT = 3;
export const RANDOM_COUNT = 3;
// Equal to the server's MAX_EXAMPLES; the server only offers "first N".
export const EXAMPLE_POOL_LIMIT = 20;

function dropDuplicateIds(items) {
  const seen = new Set();
  return items.filter((item) => !seen.has(item.id) && seen.add(item.id));
}

// Partial Fisher-Yates: moves k random items to the front of a copy of `pool`.
function drawFront(pool, count, random) {
  const arr = [...pool];
  const n = arr.length;
  for (let i = 0; i < count; i += 1) {
    const j = Math.min(n - 1, i + Math.floor(random() * (n - i)));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr.slice(0, count);
}

/**
 * items: DocumentView-like objects in server order (only `id` is read).
 * random: () => number in [0, 1). chosenId: currently chosen example id, or null.
 * Returns { defaults, random }: the first DEFAULT_COUNT items and up to
 * RANDOM_COUNT other items; the two arrays never share an id. Does not mutate items.
 */
export function pickExamples(items, { random = Math.random, chosenId = null } = {}) {
  const list = dropDuplicateIds(items);
  const defaults = list.slice(0, DEFAULT_COUNT);
  let pool = list.slice(DEFAULT_COUNT);
  const pinned = pool.filter((item) => chosenId !== null && item.id === chosenId);
  pool = pool.filter((item) => !pinned.includes(item));
  const count = Math.min(RANDOM_COUNT - pinned.length, pool.length);
  return { defaults, random: [...pinned, ...drawFront(pool, count, random)] };
}

// Foreground/background token pairs the contrast test enforces (spec: Theme tokens).
const TEXT = 4.5;
const NON_TEXT = 3;

export const TEXT_PAIRS = [
  ["--text", "--surface", TEXT],
  ["--text", "--bg", TEXT],
  ["--text-secondary", "--surface", TEXT],
  ["--text-secondary", "--bg", TEXT],
  ["--text-muted", "--surface", TEXT],
  ["--link", "--surface", TEXT],
  ["--link", "--bg", TEXT],
  ["--header-top-fg", "--header-top-bg", TEXT],
  ["--header-fg", "--header-bg", TEXT],
  ["--wordmark", "--header-bg", TEXT],
  ["--footer-fg", "--footer-bg", TEXT],
  ["--footer-link", "--footer-bg", TEXT],
  ["--btn-primary-fg", "--btn-primary-bg", TEXT],
  ["--btn-primary-fg", "--btn-primary-hover-bg", TEXT],
  ["--btn-cta-fg", "--btn-cta-bg", TEXT],
  ["--btn-secondary-fg", "--btn-secondary-bg", TEXT],
  ["--chip-shared-fg", "--chip-shared-bg", TEXT],
  ["--chip-related-fg", "--chip-related-bg", TEXT],
  ["--chip-gap-fg", "--chip-gap-bg", TEXT],
  ["--chip-other-fg", "--chip-other-bg", TEXT],
  ["--hero-fg", "--hero-bg", TEXT],
  ["--text", "--surface", TEXT], // alert text: Gray 1 on the white alert surface
];

export const NON_TEXT_PAIRS = [
  ["--border-strong", "--surface", NON_TEXT],
  ["--focus-ring", "--surface", NON_TEXT],
  ["--focus-ring", "--bg", NON_TEXT],
  ["--focus-ring-on-dark", "--footer-bg", NON_TEXT],
  ["--focus-ring-on-dark", "--hero-bg", NON_TEXT],
  ["--btn-cta-bg", "--hero-bg", NON_TEXT],
  ["--btn-secondary-bg", "--hero-bg", NON_TEXT],
];

// Combinations the brand guide forbids, as [foreground, background] color tokens.
export const FORBIDDEN_PAIRS = [
  ["--color-gold", "--color-white"],
  ["--color-white", "--color-gold"],
  ["--color-maroon", "--color-black"],
  ["--color-black", "--color-maroon"],
  ["--color-gray-1", "--color-maroon"],
];

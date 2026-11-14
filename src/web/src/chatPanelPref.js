export const PANEL_KEY = "getwork.chat.panel.v1";

const CLOSED = "closed";
const OPEN = "open";

export function loadPanelOpen() {
  try {
    return window.sessionStorage.getItem(PANEL_KEY) !== CLOSED;
  } catch {
    return true;
  }
}

export function savePanelOpen(open) {
  try {
    window.sessionStorage.setItem(PANEL_KEY, open ? OPEN : CLOSED);
  } catch {
    // Storage full or disabled: the choice lasts only until the page reloads.
  }
}

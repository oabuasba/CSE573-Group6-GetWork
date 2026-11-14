import { afterEach, expect, test, vi } from "vitest";
import { clearAllConversations, saveConversation } from "./conversationStore.js";
import { PANEL_KEY, loadPanelOpen, savePanelOpen } from "./chatPanelPref.js";

afterEach(() => {
  vi.restoreAllMocks();
});

test("defaults to open", () => {
  expect(loadPanelOpen()).toBe(true);
});

test("round trip", () => {
  savePanelOpen(false);
  expect(window.sessionStorage.getItem(PANEL_KEY)).toBe("closed");
  expect(loadPanelOpen()).toBe(false);
  savePanelOpen(true);
  expect(window.sessionStorage.getItem(PANEL_KEY)).toBe("open");
  expect(loadPanelOpen()).toBe(true);
});

test("an unknown stored value means open", () => {
  window.sessionStorage.setItem(PANEL_KEY, "banana");
  expect(loadPanelOpen()).toBe(true);
});

test("a throwing getItem means open", () => {
  vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
    throw new Error("denied");
  });
  expect(loadPanelOpen()).toBe(true);
});

test("a throwing setItem does not throw", () => {
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("full");
  });
  expect(() => savePanelOpen(false)).not.toThrow();
});

test("clearAllConversations leaves the preference", () => {
  savePanelOpen(false);
  saveConversation("m1", [{ role: "user", content: "hi" }]);
  clearAllConversations();
  expect(window.sessionStorage.getItem(PANEL_KEY)).toBe("closed");
  expect(loadPanelOpen()).toBe(false);
});

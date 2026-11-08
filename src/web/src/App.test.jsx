import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import App from "./App.jsx";
import { DISCLAIMER } from "./Layout.jsx";
import contract from "./fixtures/contract.json";

const okReply = (health) => ({ ok: true, status: 200, json: async () => health });
const liveHealth = { ...contract.health, mode: "live" };

beforeEach(() => {
  window.location.hash = "";
});
afterEach(() => vi.unstubAllGlobals());

async function renderApp(health = liveHealth) {
  const fetchImpl = vi.fn().mockResolvedValue(okReply(health));
  render(<App fetchImpl={fetchImpl} />);
  await screen.findByRole("heading", { level: 1 });
  return fetchImpl;
}

test("skip link and landmarks", async () => {
  const user = userEvent.setup();
  await renderApp();
  await user.tab();
  const skip = screen.getByRole("link", { name: "Skip to main content" });
  expect(skip).toHaveFocus();
  expect(skip).toHaveAttribute("href", "#main");
  expect(document.getElementById("main").tagName).toBe("MAIN");
  for (const role of ["banner", "navigation", "main", "contentinfo"]) {
    expect(screen.getByRole(role)).toBeInTheDocument();
  }
});

test("routing and one h1 per route", async () => {
  const user = userEvent.setup();
  await renderApp();
  const nav = screen.getByRole("navigation");
  expect(within(nav).getByRole("link", { name: "Find jobs" })).toHaveAttribute("aria-current", "page");
  for (const [name, hash] of [["Find candidates", "#/candidates"], ["About", "#/about"], ["Find jobs", "#/jobs"]]) {
    await user.click(within(nav).getByRole("link", { name }));
    expect(window.location.hash).toBe(hash);
    expect(await within(nav).findByRole("link", { name })).toHaveAttribute("aria-current", "page");
    expect(within(nav).getAllByRole("link").filter((a) => a.hasAttribute("aria-current"))).toHaveLength(1);
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  }
  await act(async () => {
    window.location.hash = "#/bogus";
  });
  expect(screen.getAllByRole("heading", { level: 1 })[0]).toHaveTextContent("Find jobs");
});

test("health failure and retry", async () => {
  const user = userEvent.setup();
  const fetchImpl = vi
    .fn()
    .mockRejectedValueOnce(new TypeError("down"))
    .mockResolvedValue(okReply(liveHealth));
  render(<App fetchImpl={fetchImpl} />);
  expect(await screen.findByText("Cannot reach the GetWork server")).toBeInTheDocument();
  expect(screen.getByText("uv run webapp serve")).toBeInTheDocument();
  expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  await user.click(screen.getByRole("button", { name: "Retry" }));
  expect(fetchImpl).toHaveBeenCalledTimes(2);
  await vi.waitFor(() => expect(screen.queryByText("Cannot reach the GetWork server")).not.toBeInTheDocument());
});

test("demo banner", async () => {
  const user = userEvent.setup();
  await renderApp(contract.health);
  for (const name of ["Find jobs", "Find candidates", "About"]) {
    await user.click(within(screen.getByRole("navigation")).getByRole("link", { name }));
    expect(screen.getByText("Demo mode.")).toBeInTheDocument();
  }
  expect(contract.health.mode).toBe("demo");
});

test("no banner outside demo mode", async () => {
  await renderApp();
  expect(screen.queryByText("Demo mode.")).not.toBeInTheDocument();
});

test("footer disclaimer", async () => {
  await renderApp();
  const footer = screen.getByRole("contentinfo");
  expect(footer).toHaveTextContent(DISCLAIMER);
  expect(DISCLAIMER).toBe(
    "GetWork is a CSE 573 course project at Arizona State University (Group 6). It is not an official ASU website.",
  );
  expect(within(footer).getByRole("link", { name: "Contact the team" })).toHaveAttribute("href", "#/about");
  expect(document.querySelectorAll("img, [role='img']")).toHaveLength(0);
});

test("fetches health once when no fetchImpl prop is given", async () => {
  const fetchMock = vi.fn().mockResolvedValue(okReply(liveHealth));
  vi.stubGlobal("fetch", fetchMock);
  render(<App />);
  await screen.findByRole("navigation");
  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(fetchMock.mock.calls[0][0]).toBe("/api/health");
});

test("real App with no props keeps network calls small across routes", async () => {
  const fetchMock = vi.fn().mockResolvedValue(okReply(liveHealth));
  vi.stubGlobal("fetch", fetchMock);
  const user = userEvent.setup();
  render(<App />);
  await screen.findByRole("navigation");
  const nav = screen.getByRole("navigation");
  for (const name of ["Find candidates", "About", "Find jobs"]) {
    await user.click(within(nav).getByRole("link", { name }));
  }
  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

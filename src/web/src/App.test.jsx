import { act, cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import App from "./App.jsx";
import Layout, { DISCLAIMER } from "./Layout.jsx";
import contract from "./fixtures/contract.json";
import agent from "./fixtures/contract_agent.json";

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

test("the shell keeps the footer last", async () => {
  for (const hash of ["#/jobs", "#/about", "#/chat/x"]) {
    window.location.hash = hash;
    const { container, unmount } = render(
      <App fetchImpl={vi.fn().mockResolvedValue(okReply(liveHealth))} />,
    );
    await screen.findByRole("heading", { level: 1 });
    expect(container.children).toHaveLength(1);
    const shell = container.firstElementChild;
    expect(shell).toHaveClass("app-shell");
    const kids = [...shell.children];
    expect(kids[0]).toHaveClass("skip-link");
    expect(kids[1].tagName).toBe("HEADER");
    expect(kids[2].tagName).toBe("MAIN");
    expect(kids[2]).toHaveClass("app-main");
    if (hash === "#/chat/x") expect(kids[2]).toHaveClass("app-main-chat");
    else expect(kids[2]).not.toHaveClass("app-main-chat");
    expect(kids[3]).toHaveClass("site-footer");
    expect(kids).toHaveLength(4);
    unmount();
  }
});

test("the shell adds the demo banner and the chat main class", () => {
  const { container } = render(
    <Layout route="chat" health={{ mode: "demo" }}>
      <h1>x</h1>
    </Layout>,
  );
  const kids = [...container.firstElementChild.children];
  expect(kids.map((el) => el.tagName)).toEqual(["A", "HEADER", "DIV", "MAIN", "FOOTER"]);
  expect(kids[3]).toHaveClass("app-main", "app-main-chat");
  expect(kids[3]).toHaveAttribute("id", "main");
  expect(kids[4]).toHaveClass("site-footer");
});

const chatHealth = { ...liveHealth, chat: { enabled: true, model: "demo-model" } };
const first = contract.match.matches[0];
const matchCalls = (fetchImpl) => fetchImpl.mock.calls.filter(([p]) => p === "/api/match");

function chatFetch() {
  return vi.fn(async (path) => {
    if (path.startsWith("/api/health")) return okReply(chatHealth);
    if (path.startsWith("/api/match")) return okReply(contract.match);
    if (path.startsWith("/api/explain")) return okReply(contract.explain);
    if (path.startsWith("/api/chat")) return okReply(agent.chat);
    return okReply(contract.examples);
  });
}

async function searchAndOpenChat(user, fetchImpl) {
  render(<App fetchImpl={fetchImpl} />);
  await user.click(await screen.findByRole("button", { name: "Fill in a form" }));
  await user.type(screen.getByLabelText("Skills"), "Vue");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  await screen.findByRole("heading", { level: 2, name: "Results" });
  await user.click(screen.getAllByRole("button", { name: "Why this match?" })[0]);
  await user.click(await screen.findByRole("link", { name: "Chat about this suggestion" }));
}

test("chat link opens the chat page", async () => {
  const user = userEvent.setup();
  await searchAndOpenChat(user, chatFetch());
  expect(window.location.hash).toBe(`#/chat/${encodeURIComponent(first.id)}`);
  expect(await screen.findByLabelText("Your question")).toBeInTheDocument();
  expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Chat about this suggestion");
  const nav = screen.getByRole("navigation");
  expect(within(nav).getAllByRole("link").filter((a) => a.hasAttribute("aria-current"))).toHaveLength(0);
  expect(screen.queryByRole("heading", { name: "Results" })).not.toBeInTheDocument();
});

test("back to results restores the card", async () => {
  const user = userEvent.setup();
  const fetchImpl = chatFetch();
  await searchAndOpenChat(user, fetchImpl);
  await user.click(await screen.findByRole("link", { name: "Back to results" }));
  expect(window.location.hash).toBe("#/jobs");
  await screen.findByRole("heading", { level: 2, name: "Results" });
  expect(screen.getAllByRole("button", { name: "Why this match?" })[0]).toHaveAttribute("aria-expanded", "true");
  expect(screen.getByRole("link", { name: "Chat about this suggestion" })).toHaveFocus();
  expect(matchCalls(fetchImpl)).toHaveLength(1);

  await act(async () => {
    window.location.hash = `#/chat/${encodeURIComponent(first.id)}`;
  });
  expect(await screen.findByLabelText("Your question")).toBeInTheDocument();
});

test("reload on the chat route uses the stored snapshot", async () => {
  const user = userEvent.setup();
  const fetchImpl = chatFetch();
  await searchAndOpenChat(user, fetchImpl);
  await user.type(await screen.findByLabelText("Your question"), "Why?");
  await user.click(screen.getByRole("button", { name: "Send" }));
  await screen.findByText(agent.chat.claims[0].text);
  cleanup();

  const reloaded = chatFetch();
  render(<App fetchImpl={reloaded} />);
  expect(await screen.findByLabelText("Your question")).toBeInTheDocument();
  expect(screen.getByRole("log")).toHaveTextContent("You: Why?");
  expect(matchCalls(reloaded)).toHaveLength(0);
});

test("chat route without a session", async () => {
  window.location.hash = "#/chat";
  render(<App fetchImpl={chatFetch()} />);
  expect(await screen.findByText(/^Open a chat from a search result/)).toBeInTheDocument();
  expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
});

test("the chat route shows its title while health is loading", () => {
  window.location.hash = "#/chat/x";
  render(<App fetchImpl={vi.fn(() => new Promise(() => {}))} />);
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Chat about this suggestion");
});

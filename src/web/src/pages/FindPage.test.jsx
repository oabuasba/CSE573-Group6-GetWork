import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import FindPage from "./FindPage.jsx";
import contract from "../fixtures/contract.json";
import agent from "../fixtures/contract_agent.json";
import { CONVERSATION_PREFIX, loadConversation, saveConversation } from "../conversationStore.js";
import { loadResults, saveResults } from "../resultsSession.js";

const health = {
  ...contract.health,
  mode: "artifacts",
  capabilities: {
    structure: { available: true, reason: null, detail: "the language model" },
    embed: { available: true, reason: null, detail: "embeddings" },
  },
};
const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const emptyMatch = { ...contract.match, matches: [], notes: [] };

afterEach(() => vi.unstubAllGlobals());

function routedFetch({ match = reply(contract.match) } = {}) {
  return vi.fn(async (path) => {
    if (path.startsWith("/api/match")) return typeof match === "function" ? match() : match;
    if (path.startsWith("/api/explain")) return reply(contract.explain);
    if (path.startsWith("/api/examples")) return reply(contract.examples);
    throw new Error(`unexpected ${path}`);
  });
}

const matchBodies = (fetchImpl) =>
  fetchImpl.mock.calls.filter(([p]) => p === "/api/match").map(([, init]) => JSON.parse(init.body));

function setup(fetchImpl, direction = "resume_to_posting") {
  const user = userEvent.setup();
  const view = render(<FindPage direction={direction} health={health} fetchImpl={fetchImpl} />);
  return { user, ...view };
}

async function fillAndSubmit(user) {
  await user.click(screen.getByRole("button", { name: "Fill in a form" }));
  await user.type(screen.getByLabelText("Skills"), "Vue, CSS");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
}

const ctaCount = (container) => container.querySelectorAll(".btn-cta").length;

test("submit, loading and results", async () => {
  let release;
  const pending = () => new Promise((resolve) => (release = () => resolve(reply(contract.match))));
  const fetchImpl = routedFetch({ match: pending });
  const { user, container } = setup(fetchImpl);
  expect(ctaCount(container)).toBe(1);
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Find jobs for your resume");
  await fillAndSubmit(user);
  expect(screen.getByRole("status")).toHaveTextContent("Finding matches...");
  expect(screen.getByRole("button", { name: "Find jobs" })).toBeDisabled();
  expect(ctaCount(container)).toBe(1);

  release();
  const heading = await screen.findByRole("heading", { level: 2, name: "Results" });
  await waitFor(() => expect(heading).toHaveFocus());
  expect(screen.getByRole("button", { name: "Find jobs" })).toBeEnabled();
  expect(screen.getByText("3 postings found")).toHaveAttribute("aria-live", "polite");
  expect(screen.getByRole("heading", { name: "What we understood" })).toBeInTheDocument();
  const note = screen.getByText(/Your document lists few skills/).closest(".alert-warning");
  expect(note).toHaveTextContent("Note");
  expect(screen.getAllByRole("listitem").filter((li) => li.querySelector("article"))).toHaveLength(3);
  expect(matchBodies(fetchImpl)[0]).toMatchObject({
    direction: "resume_to_posting", k: 10, system: "Full", query: { kind: "form", skills: ["Vue", "CSS"] },
  });
  expect(ctaCount(container)).toBe(1);
});

test("changing the ranking method resubmits with the new system", async () => {
  const fetchImpl = routedFetch();
  const { user } = setup(fetchImpl);
  await fillAndSubmit(user);
  await screen.findByRole("heading", { name: "Results" });
  await user.selectOptions(screen.getByLabelText("Ranking method"), "B1+");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  await waitFor(() => expect(matchBodies(fetchImpl)).toHaveLength(2));
  expect(matchBodies(fetchImpl)[1].system).toBe("B1+");
});

test("error states", async () => {
  const user = userEvent.setup();
  const apiError = routedFetch({ match: reply(contract.error, 503) });
  const { container, unmount } = render(<FindPage direction="resume_to_posting" health={health} fetchImpl={apiError} />);
  await user.click(screen.getByRole("button", { name: "Fill in a form" }));
  await user.type(screen.getByLabelText("Skills"), "Vue");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("That option is not available on this server");
  expect(alert).toHaveTextContent(contract.error.error.hint);
  expect(screen.getByLabelText("Skills")).toHaveValue("Vue");
  expect(ctaCount(container)).toBe(1);
  unmount();

  const down = vi.fn().mockRejectedValue(new TypeError("down"));
  render(<FindPage direction="resume_to_posting" health={health} fetchImpl={down} />);
  await user.click(screen.getByRole("button", { name: "Fill in a form" }));
  await user.type(screen.getByLabelText("Skills"), "Vue");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(/Cannot reach the GetWork server.*uv run webapp serve/);
  expect(screen.getByRole("alert").textContent.match(/Cannot reach the GetWork server/g)).toHaveLength(1);
});

test("empty state", async () => {
  const { user, container } = setup(routedFetch({ match: reply(emptyMatch) }));
  await fillAndSubmit(user);
  expect(await screen.findByRole("heading", { name: "No matches" })).toBeInTheDocument();
  expect(screen.getByText("0 postings found")).toBeInTheDocument();
  expect(ctaCount(container)).toBe(1);
});

test("explain expands a card", async () => {
  const fetchImpl = routedFetch();
  const { user, container } = setup(fetchImpl);
  await fillAndSubmit(user);
  await screen.findByRole("heading", { name: "Results" });
  const [first, second] = screen.getAllByRole("button", { name: "Why this match?" });
  await user.click(first);
  expect(await screen.findByText(contract.explain.summary)).toBeInTheDocument();
  expect(fetchImpl.mock.calls.filter(([p]) => p === "/api/explain")).toHaveLength(1);
  await user.click(second);
  await waitFor(() => expect(first).toHaveAttribute("aria-expanded", "false"));
  expect(second).toHaveAttribute("aria-expanded", "true");
  expect(ctaCount(container)).toBe(1);
});

test("one cta button per page", async () => {
  const { user, container } = setup(routedFetch(), "posting_to_resume");
  expect(ctaCount(container)).toBe(1);
  await user.click(screen.getByRole("button", { name: "Pick an example" }));
  await screen.findByRole("combobox", { name: /Example/ });
  expect(ctaCount(container)).toBe(1);
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Find candidates for your posting");
});

test("chosen example does not survive a direction change", async () => {
  const fetchImpl = routedFetch();
  const user = userEvent.setup();
  const { rerender } = render(<FindPage direction="resume_to_posting" health={health} fetchImpl={fetchImpl} />);
  await user.click(screen.getByRole("button", { name: "Pick an example" }));
  await user.selectOptions(await screen.findByRole("combobox", { name: /Example/ }), "r-1");
  rerender(<FindPage direction="posting_to_resume" health={health} fetchImpl={fetchImpl} />);
  expect(screen.getByRole("button", { name: "Paste text" })).toHaveAttribute("aria-pressed", "true");
  await user.click(screen.getByRole("button", { name: "Pick an example" }));
  expect(await screen.findByRole("combobox", { name: /Example/ })).toHaveValue("");
});

test("real FindPage without injected props makes a small number of network calls", async () => {
  const fetchImpl = routedFetch();
  vi.stubGlobal("fetch", fetchImpl);
  const user = userEvent.setup();
  render(<FindPage direction="resume_to_posting" health={health} />);
  await user.click(screen.getByRole("button", { name: "Pick an example" }));
  await user.selectOptions(await screen.findByRole("combobox", { name: /Example/ }), contract.examples.items[0].id);
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  await screen.findByRole("heading", { name: "Results" });
  await user.click(screen.getAllByRole("button", { name: "Why this match?" })[0]);
  await screen.findByText(contract.explain.summary);
  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(fetchImpl).toHaveBeenCalledTimes(3);
});

const agentHealth = { ...health, chat: { enabled: true, model: "demo-model" } };

function agentFetch({ explain = reply(agent.explain_agent) } = {}) {
  return vi.fn(async (path) => {
    if (path.startsWith("/api/match")) return reply(contract.match);
    if (path.startsWith("/api/explain")) return explain;
    if (path.startsWith("/api/chat")) return reply(agent.chat);
    if (path.startsWith("/api/examples")) return reply(contract.examples);
    throw new Error(`unexpected ${path}`);
  });
}

async function openCard(user, health_, fetchImpl, index = 0) {
  const view = render(<FindPage direction="resume_to_posting" health={health_} fetchImpl={fetchImpl} />);
  await fillAndSubmit(user);
  await screen.findByRole("heading", { name: "Results" });
  await user.click(screen.getAllByRole("button", { name: "Why this match?" })[index]);
  return view;
}

const chatLink = () => screen.queryByRole("link", { name: "Chat about this suggestion" });
const snapshotOf = () => JSON.parse(window.sessionStorage.getItem("getwork.results.v1"));

test("chat link appears only when health.chat.enabled", async () => {
  const user = userEvent.setup();
  const template = await openCard(user, health, routedFetch());
  await screen.findByText(contract.explain.summary);
  expect(chatLink()).not.toBeInTheDocument();
  template.unmount();

  const noChat = { ...health, chat: { enabled: false, reason: "off" } };
  const disabled = await openCard(user, noChat, routedFetch());
  await screen.findByText(contract.explain.summary);
  expect(chatLink()).not.toBeInTheDocument();
  disabled.unmount();

  await openCard(user, agentHealth, agentFetch());
  const link = await screen.findByRole("link", { name: "Chat about this suggestion" });
  expect(link).toHaveAttribute("href", `#/chat/${encodeURIComponent(contract.match.matches[0].id)}`);
  expect(link).toHaveClass("btn", "btn-secondary");
  expect(link).not.toHaveClass("btn-cta");
  expect(screen.getAllByRole("link", { name: "Chat about this suggestion" })).toHaveLength(1);
  expect(screen.queryByRole("heading", { name: "Ask about this match" })).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Your question")).not.toBeInTheDocument();
  expect(screen.queryByRole("log")).not.toBeInTheDocument();
  await user.click(screen.getAllByRole("button", { name: "Why this match?" })[0]);
  expect(chatLink()).not.toBeInTheDocument();
});

test("results snapshot follows the expanded card", async () => {
  const user = userEvent.setup();
  render(<FindPage direction="resume_to_posting" health={agentHealth} fetchImpl={agentFetch()} />);
  await fillAndSubmit(user);
  await screen.findByRole("heading", { name: "Results" });
  await waitFor(() => expect(snapshotOf()).toMatchObject({ direction: "resume_to_posting", expandedId: null }));
  expect(snapshotOf().response).toEqual(contract.match);
  const buttons = screen.getAllByRole("button", { name: "Why this match?" });
  await user.click(buttons[1]);
  await waitFor(() => expect(snapshotOf().expandedId).toBe(contract.match.matches[1].id));
  await user.click(buttons[1]);
  await waitFor(() => expect(snapshotOf().expandedId).toBeNull());
});

test("new search clears conversations", async () => {
  const user = userEvent.setup();
  const failing = vi.fn(async (path) => {
    if (path.startsWith("/api/match")) throw new TypeError("down");
    return reply(contract.examples);
  });
  saveConversation("j-1", [{ role: "user", content: "Hi" }]);
  const fetchImpl = agentFetch();
  const { unmount } = render(<FindPage direction="resume_to_posting" health={agentHealth} fetchImpl={fetchImpl} />);
  await fillAndSubmit(user);
  await screen.findByRole("heading", { name: "Results" });
  expect(loadConversation("j-1")).toEqual([]);
  unmount();

  saveConversation("j-1", [{ role: "user", content: "Hi" }]);
  render(<FindPage direction="resume_to_posting" health={agentHealth} fetchImpl={failing} />);
  await fillAndSubmit(user);
  await screen.findByRole("alert");
  expect(loadConversation("j-1")).toHaveLength(1);
  expect(window.sessionStorage.getItem(`${CONVERSATION_PREFIX}j-1`)).not.toBeNull();
  expect(loadResults()).not.toBeNull();
});

test("one cta button with the chat link", async () => {
  const user = userEvent.setup();
  const { container } = await openCard(user, agentHealth, agentFetch());
  await screen.findByRole("link", { name: "Chat about this suggestion" });
  expect(ctaCount(container)).toBe(1);
});

const seed = (extra = {}) =>
  saveResults({ direction: "resume_to_posting", response: contract.match, expandedId: contract.match.matches[1].id, ...extra });
const markReturning = () =>
  window.sessionStorage.setItem(
    "getwork.results.v1",
    JSON.stringify({ ...snapshotOf(), returning: true }),
  );

test("returning restores results and focus", async () => {
  seed();
  markReturning();
  const fetchImpl = agentFetch();
  render(<FindPage direction="resume_to_posting" health={agentHealth} fetchImpl={fetchImpl} />);
  expect(screen.getByRole("heading", { level: 2, name: "Results" })).toBeInTheDocument();
  const buttons = screen.getAllByRole("button", { name: "Why this match?" });
  expect(buttons[1]).toHaveAttribute("aria-expanded", "true");
  expect(buttons[0]).toHaveAttribute("aria-expanded", "false");
  expect(chatLink()).toHaveFocus();
  expect(snapshotOf().returning).toBe(false);
  await waitFor(() => expect(fetchImpl.mock.calls.filter(([p]) => p === "/api/explain")).toHaveLength(1));
  expect(fetchImpl.mock.calls.some(([p]) => p === "/api/match")).toBe(false);
});

test("returning tolerates an unknown expanded id", () => {
  seed({ expandedId: { odd: true } });
  markReturning();
  render(<FindPage direction="resume_to_posting" health={agentHealth} fetchImpl={agentFetch()} />);
  expect(screen.getByRole("heading", { level: 2, name: "Results" })).toBeInTheDocument();
  for (const b of screen.getAllByRole("button", { name: "Why this match?" })) {
    expect(b).toHaveAttribute("aria-expanded", "false");
  }
});

test("returning ignores a snapshot with null entries in matches", () => {
  seed();
  const stored = snapshotOf();
  window.sessionStorage.setItem(
    "getwork.results.v1",
    JSON.stringify({ ...stored, returning: true, response: { ...stored.response, matches: [null] } }),
  );
  render(<FindPage direction="resume_to_posting" health={agentHealth} fetchImpl={agentFetch()} />);
  expect(screen.queryByRole("heading", { name: "Results" })).not.toBeInTheDocument();
});

test("no restore without the flag or for the other direction", () => {
  seed();
  const first = render(<FindPage direction="resume_to_posting" health={agentHealth} fetchImpl={agentFetch()} />);
  expect(screen.queryByRole("heading", { name: "Results" })).not.toBeInTheDocument();
  first.unmount();

  markReturning();
  const other = render(<FindPage direction="posting_to_resume" health={agentHealth} fetchImpl={agentFetch()} />);
  expect(screen.queryByRole("heading", { name: "Results" })).not.toBeInTheDocument();
  other.unmount();

  render(<FindPage direction="resume_to_posting" health={agentHealth} fetchImpl={agentFetch()} />);
  expect(screen.getByRole("heading", { name: "Results" })).toBeInTheDocument();
  expect(snapshotOf().returning).toBe(false);
});

test("a second visit after the flag was consumed starts empty", () => {
  seed();
  markReturning();
  render(<FindPage direction="resume_to_posting" health={agentHealth} fetchImpl={agentFetch()} />).unmount();
  render(<FindPage direction="resume_to_posting" health={agentHealth} fetchImpl={agentFetch()} />);
  expect(screen.queryByRole("heading", { name: "Results" })).not.toBeInTheDocument();
});

test("new search after a restore", async () => {
  seed();
  markReturning();
  const user = userEvent.setup();
  const fetchImpl = agentFetch();
  render(<FindPage direction="resume_to_posting" health={agentHealth} fetchImpl={fetchImpl} />);
  await fillAndSubmit(user);
  await waitFor(() => expect(fetchImpl.mock.calls.filter(([p]) => p === "/api/match")).toHaveLength(1));
  await screen.findByRole("heading", { name: "Results" });
  await waitFor(() => expect(snapshotOf().expandedId).toBeNull());
  expect(snapshotOf().returning).toBe(false);
});

test("evidence stays when the assistant fails", async () => {
  const user = userEvent.setup();
  await openCard(user, agentHealth, agentFetch({ explain: reply(agent.chat_error, 502) }));
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("The assistant could not answer");
  expect(alert).toHaveTextContent(agent.chat_error.error.message);
  expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  const card = alert.closest("article");
  expect(card.querySelector(".chip")).not.toBeNull();
});

async function showResults(user) {
  await fillAndSubmit(user);
  await screen.findByRole("heading", { level: 2, name: "Results" });
}
const toggle = (name) => screen.getByRole("button", { name });
const understoodPanel = () => document.getElementById(toggle(/details$/).getAttribute("aria-controls"));

test("understood card is closed by default", async () => {
  const { user } = setup(routedFetch());
  await showResults(user);
  expect(screen.getByRole("heading", { name: "What we understood" })).toBeInTheDocument();
  expect(toggle("Show details")).toHaveAttribute("aria-expanded", "false");
  expect(understoodPanel()).toBeEmptyDOMElement();
  expect(screen.queryByText(/years of experience/)).not.toBeInTheDocument();
});

test("understood card opens, shows the same content and toggles closed", async () => {
  const { user } = setup(routedFetch());
  await showResults(user);
  await user.click(toggle("Show details"));
  const button = toggle("Hide details");
  expect(button).toHaveAttribute("aria-expanded", "true");
  expect(button).toHaveAttribute("aria-controls", understoodPanel().id);
  const panel = understoodPanel();
  expect(panel).toBeVisible();
  expect(panel).toHaveTextContent("Frontend Developer, Associate, 3 years of experience");
  expect(panel).toHaveTextContent("Front-end");
  await user.click(button);
  expect(toggle("Show details")).toHaveAttribute("aria-expanded", "false");
  expect(screen.queryByText(/years of experience/)).not.toBeInTheDocument();
});

test("understood card resets to closed when new results arrive", async () => {
  const { user } = setup(routedFetch());
  await showResults(user);
  await user.click(toggle("Show details"));
  expect(toggle("Hide details")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  await screen.findByRole("heading", { level: 2, name: "Results" });
  expect(toggle("Show details")).toHaveAttribute("aria-expanded", "false");
});

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "vitest-axe";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import App from "./App.jsx";
import ChatPage from "./pages/ChatPage.jsx";
import FindPage from "./pages/FindPage.jsx";
import { saveResults } from "./resultsSession.js";
import contract from "./fixtures/contract.json";
import agent from "./fixtures/contract_agent.json";

// jsdom cannot compute colors, so contrast is covered by the token test instead (AC-27).
const AXE_OPTIONS = { rules: { "color-contrast": { enabled: false } } };
const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const health = {
  ...contract.health,
  capabilities: {
    structure: { available: true, reason: null, detail: "the language model" },
    embed: { available: true, reason: null, detail: "embeddings" },
  },
};

function fakeFetch(matchReply) {
  return vi.fn(async (path) => {
    if (path.startsWith("/api/health")) return reply(contract.health);
    if (path.startsWith("/api/examples")) return reply(contract.examples);
    if (path.startsWith("/api/match")) return matchReply;
    return reply(contract.explain);
  });
}

beforeEach(() => {
  window.location.hash = "";
});
afterEach(() => vi.unstubAllGlobals());

test.each(["#/home", "#/jobs", "#/candidates", "#/about"])("no axe violations on each route %s", async (hash) => {
  window.location.hash = hash;
  const { container } = render(<App fetchImpl={fakeFetch(reply(contract.match))} />);
  await screen.findByRole("heading", { level: 1 });
  await vi.waitFor(() => expect(screen.getByRole("main")).not.toHaveTextContent("Loading"));
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("home page structure", async () => {
  window.location.hash = "#/home";
  render(<App fetchImpl={fakeFetch(reply(contract.match))} />);
  await screen.findByRole("heading", { level: 1, name: "Match resumes to jobs, and jobs to resumes" });
  const main = screen.getByRole("main");
  const levels = [...main.querySelectorAll("h1, h2, h3, h4, h5, h6")].map((h) => Number(h.tagName[1]));
  expect(levels).toEqual([1, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 2]);
  const sections = [...main.querySelectorAll("section")];
  expect(sections.length).toBeGreaterThan(0);
  for (const section of sections) {
    const labelId = section.getAttribute("aria-labelledby");
    expect(labelId).toBeTruthy();
    expect(document.getElementById(labelId)).not.toBeNull();
  }
  const buttons = [...main.querySelectorAll(".btn")];
  expect(buttons.length).toBeGreaterThan(0);
  for (const el of buttons) {
    expect(el.tagName).toBe("A");
    expect(el).toHaveAttribute("href");
  }
});

async function searchAndAxe(matchReply, expand = false) {
  const user = userEvent.setup();
  const { container } = render(
    <FindPage direction="resume_to_posting" health={health} fetchImpl={fakeFetch(matchReply)} />,
  );
  await user.click(screen.getByRole("button", { name: "Fill in a form" }));
  await user.type(screen.getByLabelText("Skills"), "Vue");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  if (matchReply.ok) await screen.findByRole("heading", { name: "Results" });
  else await screen.findByRole("alert");
  if (expand) {
    await user.click(screen.getAllByRole("button", { name: "Why this match?" })[0]);
    await screen.findByText(contract.explain.summary);
  }
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
}

test("no axe violations in result and error states", async () => {
  await searchAndAxe(reply(contract.match));
});

test("no axe violations with the understood card open", async () => {
  const user = userEvent.setup();
  const { container } = render(
    <FindPage direction="resume_to_posting" health={health} fetchImpl={fakeFetch(reply(contract.match))} />,
  );
  await user.click(screen.getByRole("button", { name: "Fill in a form" }));
  await user.type(screen.getByLabelText("Skills"), "Vue");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  await screen.findByRole("heading", { name: "Results" });
  await user.click(screen.getByRole("button", { name: "Show details" }));
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("no axe violations with an error alert", async () => {
  await searchAndAxe(reply(contract.error, 503));
});

test("no axe violations with an expanded explanation", async () => {
  await searchAndAxe(reply(contract.match), true);
});

const agentHealth = { ...health, chat: { enabled: true, model: "demo-model" } };

async function openAgentCard(explainReply) {
  const user = userEvent.setup();
  const fetchImpl = vi.fn(async (path) => {
    if (path.startsWith("/api/match")) return reply(contract.match);
    if (path.startsWith("/api/explain")) return explainReply;
    if (path.startsWith("/api/chat")) return reply(agent.chat_error, 502);
    return reply(contract.examples);
  });
  const view = render(<FindPage direction="resume_to_posting" health={agentHealth} fetchImpl={fetchImpl} />);
  await user.click(screen.getByRole("button", { name: "Fill in a form" }));
  await user.type(screen.getByLabelText("Skills"), "Vue");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  await screen.findByRole("heading", { name: "Results" });
  await user.click(screen.getAllByRole("button", { name: "Why this match?" })[0]);
  return { user, ...view };
}

test("no axe violations on the find page with the example select", async () => {
  const user = userEvent.setup();
  const { container } = render(
    <FindPage direction="resume_to_posting" health={health} fetchImpl={fakeFetch(reply(contract.match))} />,
  );
  await user.click(screen.getByRole("button", { name: "Pick an example" }));
  await screen.findByRole("combobox", { name: /Example/ });
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("no axe violations on the find page with grouped examples", async () => {
  const items = Array.from({ length: 10 }, (_, i) => ({ ...contract.examples.items[0], id: `r-${i + 1}`, title: `Resume ${i + 1}` }));
  const fetchImpl = vi.fn(async (path) => {
    if (path.startsWith("/api/examples")) return reply({ items });
    return reply(contract.match);
  });
  const user = userEvent.setup();
  const { container } = render(<FindPage direction="resume_to_posting" health={health} fetchImpl={fetchImpl} />);
  await user.click(screen.getByRole("button", { name: "Pick an example" }));
  const select = await screen.findByRole("combobox", { name: /Example/ });
  const groups = [...select.querySelectorAll("optgroup")];
  expect(groups).toHaveLength(2);
  for (const group of groups) expect(group.getAttribute("label")?.trim()).toBeTruthy();
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("no axe violations with the chat link", async () => {
  const { container, user } = await openAgentCard(reply(contract.explain));
  await screen.findByRole("link", { name: "Chat about this suggestion" });
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
  expect(user).toBeDefined();
});

const chatHealth = { ...health, chat: { enabled: true, model: "demo-model" } };
const focus = contract.match.matches[0];

function renderChat({ chat, health: h = chatHealth, id = focus.id, seed = true } = {}) {
  if (seed) saveResults({ direction: "resume_to_posting", response: contract.match, expandedId: null });
  const fetchImpl = vi.fn(async (path) => {
    if (path.startsWith("/api/chat")) return chat();
    return reply(contract.explain);
  });
  const user = userEvent.setup();
  return { user, ...render(<ChatPage matchId={id} health={h} fetchImpl={fetchImpl} />) };
}

async function sendMessage(user) {
  await user.type(screen.getByLabelText("Your question"), "Why?");
  await user.click(screen.getByRole("button", { name: "Send" }));
}

const headingLevels = () => screen.getAllByRole("heading").map((h) => Number(h.tagName[1]));

test("no axe violations on the chat page empty", async () => {
  const { container } = renderChat({ chat: () => reply(agent.chat) });
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
  expect(headingLevels()).toEqual([1, 2, 2, 3]);
});

test("no axe violations on the chat page with the suggestion hidden", async () => {
  const { container, user } = renderChat({ chat: () => reply(agent.chat) });
  await user.click(screen.getByRole("button", { name: "Hide suggestion" }));
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("no axe violations on the chat page with text in the composer", async () => {
  const { container, user } = renderChat({ chat: () => reply(agent.chat) });
  await user.type(screen.getByLabelText("Your question"), "Why is this ranked here?");
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("no axe violations on the chat page with turns", async () => {
  const { container, user } = renderChat({ chat: () => reply(agent.chat) });
  await sendMessage(user);
  await screen.findByText((_, el) => el?.textContent === agent.chat.reply && el.classList.contains("chat-reply"));
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("no axe violations on the chat page while pending", async () => {
  const { container, user } = renderChat({ chat: () => new Promise(() => {}) });
  await sendMessage(user);
  await screen.findByRole("status");
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("no axe violations on the chat page with an error", async () => {
  const { container, user } = renderChat({ chat: () => reply(agent.chat_error, 502) });
  await sendMessage(user);
  await screen.findByRole("alert");
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("no axe violations on the chat page when chat is disabled", async () => {
  const off = { ...health, chat: { enabled: false, reason: "Set VOYAGER_API_KEY to enable chat." } };
  const { container } = renderChat({ chat: () => reply(agent.chat), health: off });
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("no axe violations on the chat page without a session", async () => {
  const { container } = renderChat({ chat: () => reply(agent.chat), seed: false });
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

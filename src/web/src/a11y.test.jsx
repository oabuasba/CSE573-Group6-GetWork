import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "vitest-axe";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import App from "./App.jsx";
import FindPage from "./pages/FindPage.jsx";
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

test.each(["#/jobs", "#/candidates", "#/about"])("no axe violations on each route %s", async (hash) => {
  window.location.hash = hash;
  const { container } = render(<App fetchImpl={fakeFetch(reply(contract.match))} />);
  await screen.findByRole("heading", { level: 1 });
  await vi.waitFor(() => expect(screen.getByRole("main")).not.toHaveTextContent("Loading"));
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
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

const agentHealth = { ...health, explainer: { kind: "agent", model: "demo-model" } };

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

test("no axe violations with chat open", async () => {
  const { container, user } = await openAgentCard(reply(agent.explain_agent));
  await screen.findByText(agent.explain_agent.claims[0].text);
  await user.click(screen.getByRole("button", { name: "Chat about this suggestion" }));
  await screen.findByRole("heading", { name: "Ask about this match" });
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("no axe violations with the summary only", async () => {
  const { container } = await openAgentCard(reply(agent.explain_agent));
  await screen.findByText(agent.explain_agent.claims[0].text);
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("no axe violations with all points expanded", async () => {
  const { container, user } = await openAgentCard(reply(agent.explain_agent));
  await user.click(await screen.findByRole("button", { name: /^Show all \d+ points$/ }));
  await user.click(screen.getAllByRole("button", { name: "Show evidence" })[0]);
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

test("no axe violations with an assistant error", async () => {
  const { container, user } = await openAgentCard(reply(agent.chat_error, 502));
  await user.click(await screen.findByRole("button", { name: "Chat about this suggestion" }));
  await user.type(await screen.findByLabelText("Your question"), "Why?");
  await user.click(screen.getByRole("button", { name: "Send" }));
  expect(await screen.findAllByRole("alert")).toHaveLength(2);
  expect(container.querySelectorAll(".btn-cta").length).toBeLessThanOrEqual(1);
  expect(await axe(container, AXE_OPTIONS)).toHaveNoViolations();
});

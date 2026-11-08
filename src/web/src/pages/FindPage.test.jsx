import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import FindPage from "./FindPage.jsx";
import contract from "../fixtures/contract.json";

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
  await screen.findAllByRole("radio");
  expect(ctaCount(container)).toBe(1);
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Find candidates for your posting");
});

test("chosen example does not survive a direction change", async () => {
  const fetchImpl = routedFetch();
  const user = userEvent.setup();
  const { rerender } = render(<FindPage direction="resume_to_posting" health={health} fetchImpl={fetchImpl} />);
  await user.click(screen.getByRole("button", { name: "Pick an example" }));
  await user.click((await screen.findAllByRole("radio"))[0]);
  rerender(<FindPage direction="posting_to_resume" health={health} fetchImpl={fetchImpl} />);
  expect(screen.getByRole("button", { name: "Paste text" })).toHaveAttribute("aria-pressed", "true");
  await user.click(screen.getByRole("button", { name: "Pick an example" }));
  for (const radio of await screen.findAllByRole("radio")) expect(radio).not.toBeChecked();
});

test("real FindPage without injected props makes a small number of network calls", async () => {
  const fetchImpl = routedFetch();
  vi.stubGlobal("fetch", fetchImpl);
  const user = userEvent.setup();
  render(<FindPage direction="resume_to_posting" health={health} />);
  await user.click(screen.getByRole("button", { name: "Pick an example" }));
  await user.click((await screen.findAllByRole("radio"))[0]);
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  await screen.findByRole("heading", { name: "Results" });
  await user.click(screen.getAllByRole("button", { name: "Why this match?" })[0]);
  await screen.findByText(contract.explain.summary);
  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(fetchImpl).toHaveBeenCalledTimes(3);
});

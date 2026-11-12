import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StrictMode } from "react";
import { expect, test, vi } from "vitest";
import ExplainPanel from "./ExplainPanel.jsx";
import contract from "../fixtures/contract.json";
import agent from "../fixtures/contract_agent.json";

const query = contract.match.query;
const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });

function renderPanel(fetchImpl) {
  return render(
    <StrictMode>
      <ExplainPanel direction="resume_to_posting" query={query} matchId="j-1" fetchImpl={fetchImpl} />
    </StrictMode>,
  );
}

test("loads and renders an explanation", async () => {
  const fetchImpl = vi.fn().mockResolvedValue(reply(contract.explain));
  renderPanel(fetchImpl);
  expect(screen.getByRole("status")).toHaveTextContent("Loading the explanation...");
  expect(await screen.findByText(contract.explain.summary)).toBeInTheDocument();
  for (const bullet of contract.explain.bullets) expect(screen.getByText(bullet)).toBeInTheDocument();
  expect(screen.getByText("Explained by rules, no language model")).toBeInTheDocument();
  expect(fetchImpl).toHaveBeenCalledTimes(1);
  expect(fetchImpl.mock.calls[0][0]).toBe("/api/explain");
  expect(JSON.parse(fetchImpl.mock.calls[0][1].body)).toEqual({
    direction: "resume_to_posting", query, match_id: "j-1",
  });
});

test("names the model for an agent explainer", async () => {
  const agent = { ...contract.explain, explainer: { kind: "agent", model: "model-x" } };
  renderPanel(vi.fn().mockResolvedValue(reply(agent)));
  expect(await screen.findByText("Explained by model-x")).toBeInTheDocument();
});

test("shows an error and retries", async () => {
  const user = userEvent.setup();
  const fetchImpl = vi
    .fn()
    .mockResolvedValueOnce(reply(contract.error, 503))
    .mockResolvedValue(reply(contract.explain));
  renderPanel(fetchImpl);
  expect(await screen.findByRole("alert")).toHaveTextContent("That option is not available on this server");
  expect(screen.getByRole("alert")).toHaveTextContent(contract.error.error.hint);
  await user.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByText(contract.explain.summary)).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(fetchImpl).toHaveBeenCalledTimes(2);
});

test("agent explanation shows the summary and hides the extra points", async () => {
  renderPanel(vi.fn().mockResolvedValue(reply(agent.explain_agent)));
  const [first, ...rest] = agent.explain_agent.claims;
  expect(await screen.findByText(first.text)).toBeVisible();
  for (const claim of rest) expect(screen.queryByText(claim.text)).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Show evidence" })).not.toBeInTheDocument();
  const toggle = screen.getByRole("button", { name: `Show all ${agent.explain_agent.claims.length} points` });
  expect(toggle).toHaveAttribute("aria-expanded", "false");
  expect(document.getElementById(toggle.getAttribute("aria-controls"))).toBeInTheDocument();
  expect(screen.getByText("Explained by demo-model")).toBeInTheDocument();
});

test("the toggle shows all points with working evidence buttons", async () => {
  const user = userEvent.setup();
  const { claims, facts } = agent.explain_agent;
  renderPanel(vi.fn().mockResolvedValue(reply(agent.explain_agent)));
  const toggle = await screen.findByRole("button", { name: `Show all ${claims.length} points` });
  await user.click(toggle);
  expect(document.activeElement).toBe(toggle);
  for (const claim of claims) expect(screen.getByText(claim.text)).toBeVisible();
  const buttons = screen.getAllByRole("button", { name: "Show evidence" });
  expect(buttons).toHaveLength(claims.length);
  await user.click(buttons[0]);
  expect(screen.getByText(facts[0].text)).toBeInTheDocument();
  const hide = screen.getByRole("button", { name: "Show less" });
  expect(hide).toHaveAttribute("aria-expanded", "true");
  await user.click(hide);
  expect(screen.queryByText(claims[1].text)).not.toBeInTheDocument();
  expect(screen.getByText(claims[0].text)).toBeVisible();
});

test("shows the assistant error with retry", async () => {
  const user = userEvent.setup();
  const { message, hint } = agent.chat_error.error;
  const fetchImpl = vi
    .fn()
    .mockResolvedValueOnce(reply(agent.chat_error, 502))
    .mockResolvedValue(reply(agent.explain_agent));
  renderPanel(fetchImpl);
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("The assistant could not answer");
  expect(alert).toHaveTextContent(message);
  expect(alert).toHaveTextContent(hint);
  await user.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByText(agent.explain_agent.claims[0].text)).toBeInTheDocument();
});

test("a single claim shows its text and evidence without a toggle", async () => {
  const { claims, facts } = agent.explain_agent;
  const one = { ...agent.explain_agent, claims: claims.slice(0, 1), summary: claims[0].text, bullets: [], facts };
  renderPanel(vi.fn().mockResolvedValue(reply(one)));
  expect(await screen.findByText(claims[0].text)).toBeVisible();
  expect(screen.getAllByRole("button", { name: "Show evidence" })).toHaveLength(1);
  expect(screen.queryByRole("button", { name: /Show all/ })).not.toBeInTheDocument();
});

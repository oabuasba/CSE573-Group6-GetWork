import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { StrictMode } from "react";
import { expect, test, vi } from "vitest";
import ExplainPanel from "./ExplainPanel.jsx";
import contract from "../fixtures/contract.json";

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

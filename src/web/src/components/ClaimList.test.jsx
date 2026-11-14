import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";
import ClaimList from "./ClaimList.jsx";
import agent from "../fixtures/contract_agent.json";

test("lists claims and toggles evidence", async () => {
  const user = userEvent.setup();
  const { claims, facts } = agent.explain_agent;
  render(<ClaimList claims={claims} facts={facts} />);
  const list = document.querySelector("ol");
  expect(list).not.toBeNull();
  expect(list.querySelectorAll(":scope > li")).toHaveLength(claims.length);
  for (const claim of claims) expect(screen.getByText(claim.text)).toBeInTheDocument();
  const buttons = screen.getAllByRole("button", { name: "Show evidence" });
  expect(buttons).toHaveLength(claims.length);
  expect(buttons[0]).toHaveAttribute("aria-expanded", "false");
  expect(screen.queryByText(facts[0].text)).not.toBeInTheDocument();

  await user.click(buttons[0]);
  expect(buttons[0]).toHaveAttribute("aria-expanded", "true");
  expect(buttons[0]).toHaveTextContent("Hide evidence");
  expect(screen.getByText(facts[0].text)).toBeInTheDocument();
  expect(screen.getByText(`Skills: ${claims[0].skills.join(", ")}`)).toBeInTheDocument();

  await user.click(buttons[0]);
  expect(buttons[0]).toHaveAttribute("aria-expanded", "false");
  expect(screen.queryByText(facts[0].text)).not.toBeInTheDocument();
});

test("ignores unknown fact ids", async () => {
  const user = userEvent.setup();
  const claims = [{ text: "T", doc_ids: [], skills: [], facts: ["nope", "f1"] }];
  render(<ClaimList claims={claims} facts={[{ id: "f1", text: "Known fact" }]} />);
  await user.click(screen.getByRole("button", { name: "Show evidence" }));
  expect(screen.getByText("Known fact")).toBeInTheDocument();
});

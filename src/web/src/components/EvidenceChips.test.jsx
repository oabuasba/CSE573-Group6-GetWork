import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import EvidenceChips from "./EvidenceChips.jsx";
import contract from "../fixtures/contract.json";

const { evidence } = contract.match.matches[0];

test("labels and symbols", () => {
  render(<EvidenceChips evidence={evidence} />);
  for (const label of ["Shared", "Related", "Missing"]) expect(screen.getByText(label)).toBeVisible();

  const shared = screen.getByText("JavaScript", { selector: "li" });
  expect(shared).toHaveClass("chip-shared");
  expect(shared.querySelector("span")).toHaveTextContent("✓");
  expect(shared.querySelector("span")).toHaveAttribute("aria-hidden", "true");

  const related = screen.getByText("Vue and React in Front-end", { selector: "li" });
  expect(related).toHaveClass("chip-related");
  expect(related.querySelector("span")).toHaveTextContent("≈");

  const gap = screen.getByText("Kubernetes", { selector: "li" });
  expect(gap).toHaveClass("chip-gap");
  expect(gap.querySelector("span")).toHaveTextContent("✗");
});

test("renders nothing for an empty kind", () => {
  render(<EvidenceChips evidence={{ ...evidence, gaps: [], related: [] }} />);
  expect(screen.getByText("Shared")).toBeInTheDocument();
  expect(screen.queryByText("Related")).not.toBeInTheDocument();
  expect(screen.queryByText("Missing")).not.toBeInTheDocument();
});

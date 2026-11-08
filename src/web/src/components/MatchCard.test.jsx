import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import MatchCard from "./MatchCard.jsx";
import contract from "../fixtures/contract.json";

const match = contract.match.matches[0];

test("card content and toggle", async () => {
  const user = userEvent.setup();
  const onToggle = vi.fn();
  const { rerender } = render(
    <MatchCard match={match} system="Full" expanded={false} onToggle={onToggle}>
      <p>Details here</p>
    </MatchCard>,
  );
  expect(screen.getByText("Rank 1")).toBeInTheDocument();
  expect(screen.getByRole("heading", { level: 3, name: "Frontend Engineer" })).toBeInTheDocument();
  expect(screen.getByText("Associate")).toBeInTheDocument();
  expect(screen.getByText("0.812")).toBeInTheDocument();
  expect(screen.getByText("Similarity")).toBeInTheDocument();
  expect(screen.getByText("JavaScript", { selector: "li" })).toHaveClass("chip-shared");

  const button = screen.getByRole("button", { name: "Why this match?" });
  expect(button).toHaveAttribute("aria-expanded", "false");
  expect(screen.queryByText("Details here")).not.toBeInTheDocument();
  await user.click(button);
  expect(onToggle).toHaveBeenCalledTimes(1);

  rerender(
    <MatchCard match={match} system="Full" expanded onToggle={onToggle}>
      <p>Details here</p>
    </MatchCard>,
  );
  expect(screen.getByRole("button", { name: "Why this match?" })).toHaveAttribute("aria-expanded", "true");
  expect(screen.getByText("Details here")).toBeVisible();
});

test("keyword score label and untitled fallback", () => {
  const untitled = { ...match, document: { ...match.document, title: null } };
  render(<MatchCard match={untitled} system="B1+" expanded={false} onToggle={() => {}} />);
  expect(screen.getByText("Keyword score")).toBeInTheDocument();
  expect(screen.getByRole("heading", { level: 3, name: "Untitled" })).toBeInTheDocument();
});

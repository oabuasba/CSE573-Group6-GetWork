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

const candidate = (display_name) => ({
  ...match,
  display_name,
  document: { ...match.document, side: "resume", title: "Java Developer", level: "Associate" },
});

test("candidate card shows name, tag, title, level and contact line", async () => {
  const user = userEvent.setup();
  const onToggle = vi.fn();
  const { container } = render(
    <MatchCard match={candidate("Priya Nair")} system="Full" expanded={false} onToggle={onToggle} />,
  );
  expect(screen.getByRole("heading", { level: 3, name: "Priya Nair" })).toBeInTheDocument();
  expect(screen.getByText("Demo name")).toHaveAttribute("data-demo-name");
  expect(screen.getByText("Java Developer").tagName).toBe("P");
  expect(screen.getByText("Associate").tagName).toBe("P");
  expect(screen.getByText("Contact not available in dataset").tagName).toBe("P");
  expect(screen.getByText("0.812")).toBeInTheDocument();
  expect(container.querySelector("article")).toHaveAttribute("data-candidate");
  await user.click(screen.getByRole("button", { name: "Why this match?" }));
  expect(onToggle).toHaveBeenCalledTimes(1);
});

test.each([null, undefined, "  "])("candidate card without a name falls back", (name) => {
  render(<MatchCard match={candidate(name)} system="Full" expanded={false} onToggle={() => {}} />);
  expect(screen.getByRole("heading", { level: 3, name: "Unnamed candidate" })).toBeInTheDocument();
  expect(screen.queryByText("Demo name")).not.toBeInTheDocument();
  expect(screen.getByText("Java Developer")).toBeInTheDocument();
  expect(screen.getByText("Associate")).toBeInTheDocument();
  expect(screen.getByText("Contact not available in dataset")).toBeInTheDocument();
});

test("job card has no candidate extras", () => {
  const { container } = render(<MatchCard match={match} system="Full" expanded={false} onToggle={() => {}} />);
  expect(screen.queryByText("Demo name")).not.toBeInTheDocument();
  expect(screen.queryByText("Contact not available in dataset")).not.toBeInTheDocument();
  expect(container.querySelector("[data-candidate]")).toBeNull();
});

test("candidate card shows no contact details", () => {
  const { container } = render(
    <MatchCard match={candidate("Priya Nair")} system="Full" expanded={false} onToggle={() => {}} />,
  );
  expect(container.querySelector("a")).toBeNull();
  expect(container.querySelector('a[href^="mailto:"], a[href^="tel:"]')).toBeNull();
  expect(container.textContent).not.toMatch(/\S+@\S+\.\S+/);
  expect(container.textContent).not.toMatch(/\d{3}[-. ]\d{3}[-. ]\d{4}/);
});

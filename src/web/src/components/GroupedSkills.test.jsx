import { render, screen, within } from "@testing-library/react";
import { expect, test } from "vitest";
import GroupedSkills from "./GroupedSkills.jsx";

const skill = (name, cluster) => ({ skill: name, cluster, via: "taxonomy" });

test("groups and orders", () => {
  render(
    <GroupedSkills
      skills={[skill("SQL", "Other"), skill("Vue", "Front-end"), skill("Docker", "DevOps"), skill("CSS", "Front-end")]}
    />,
  );
  const headings = screen.getAllByRole("heading", { level: 4 }).map((h) => h.textContent);
  expect(headings).toEqual(["Front-end", "DevOps", "Other"]);
  const lists = screen.getAllByRole("list");
  expect(within(lists[0]).getAllByRole("listitem").map((li) => li.textContent)).toEqual(["Vue", "CSS"]);
  expect(within(lists[2]).getByText("SQL")).toHaveClass("chip-other");
  expect(within(lists[0]).getByText("Vue")).not.toHaveClass("chip-other");
});

test("renders nothing for no skills", () => {
  render(<GroupedSkills skills={[]} />);
  expect(screen.queryByRole("heading")).not.toBeInTheDocument();
});

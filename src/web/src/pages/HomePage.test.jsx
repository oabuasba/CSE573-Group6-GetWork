import { render, screen, within } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import HomePage from "./HomePage.jsx";

const STEP_TITLES = [
  "Share a resume or a posting",
  "GetWork reads it",
  "It finds the closest matches",
  "You see why, and can ask more",
];
const BENEFIT_TITLES = [
  "Both directions",
  "Related skills earn partial credit",
  "Reasons you can read",
  "Choose how matches are scored",
  "Ask follow-up questions",
  "Nothing kept on the server",
];

afterEach(() => vi.unstubAllGlobals());

const titlesIn = (list) => within(list).getAllByRole("heading", { level: 3 }).map((h) => h.textContent);

test("hero has the value proposition and two calls to action", () => {
  render(<HomePage />);
  const hero = screen.getByRole("region", { name: "Match resumes to jobs, and jobs to resumes" });
  expect(within(hero).getAllByRole("heading", { level: 1 })).toHaveLength(1);
  expect(hero.querySelector(".home-lead").textContent).toBe(
    "GetWork ranks job postings for a resume, or candidates for a posting, by comparing skills, job titles and experience levels. Its default method gives partial credit to related skills, such as React and Vue, and every match comes with the reasons behind it.",
  );
  const links = within(hero).getAllByRole("link");
  expect(links.map((link) => link.textContent)).toEqual(["Find jobs for my resume", "Find candidates for a posting"]);
  expect(links[0]).toHaveAttribute("href", "#/jobs");
  expect(links[0]).toHaveClass("btn", "btn-cta");
  expect(links[1]).toHaveAttribute("href", "#/candidates");
  expect(links[1]).toHaveClass("btn", "btn-secondary");
  expect(hero.querySelector(".home-hint").textContent).toBe(
    'No resume or posting at hand? Choose "Pick an example" on the next page.',
  );
});

test("how it works has four steps in order", () => {
  render(<HomePage />);
  const heading = screen.getByRole("heading", { level: 2, name: "How it works" });
  const list = heading.closest("section").querySelector("ol");
  expect(within(list).getAllByRole("listitem")).toHaveLength(4);
  expect(titlesIn(list)).toEqual(STEP_TITLES);
  const text = list.textContent;
  expect(text).toContain("Paste the text, fill in a short form, or pick an example.");
  expect(text).toContain("so JS and JavaScript count as one");
  expect(text).toContain("lists the highest scores first");
  expect(text).toContain("by rules");
  expect(text).toContain("Where the chat agent is switched on");
  expect(text).toContain("checked against the two documents");
});

test("what you get lists six benefits", () => {
  render(<HomePage />);
  const heading = screen.getByRole("heading", { level: 2, name: "What you get" });
  const list = heading.closest("section").querySelector("ul");
  const items = within(list).getAllByRole("listitem");
  expect(items).toHaveLength(6);
  expect(titlesIn(list)).toEqual(BENEFIT_TITLES);
  expect(within(items[3]).getByRole("link", { name: "About page" })).toHaveAttribute("href", "#/about");
  expect(items[4].textContent).toContain("Where the chat agent is switched on");
});

test("says what it is and is not", () => {
  render(<HomePage />);
  const section = screen.getByRole("heading", { level: 2, name: "What GetWork is, and is not" }).closest("section");
  for (const phrase of [
    "course project",
    "not an official ASU website",
    "not a job board",
    "fixed set of resumes and job postings",
    "not against live listings",
    "has not been formally evaluated yet",
  ]) {
    expect(section.textContent).toContain(phrase);
  }
  expect(within(section).getByRole("link", { name: "About page" })).toHaveAttribute("href", "#/about");
});

test("closing call to action", () => {
  render(<HomePage />);
  const section = screen.getByRole("heading", { level: 2, name: "Ready to see your matches?" }).closest("section");
  const primary = within(section).getByRole("link", { name: "Try it with your resume" });
  const secondary = within(section).getByRole("link", { name: "Try it with a job posting" });
  expect(primary).toHaveAttribute("href", "#/jobs");
  expect(primary).toHaveClass("btn-primary");
  expect(secondary).toHaveAttribute("href", "#/candidates");
  expect(secondary).toHaveClass("btn-secondary");
});

test("makes no claims it cannot back", () => {
  const { container } = render(<HomePage />);
  const claims =
    /\b(best|most accurate|accurate|accuracy|guarantee\w*|proven|state[- ]of[- ]the[- ]art|ai[- ]powered|instant\w*|real[- ]time|thousands|millions|outperform\w*|better than)\b/i;
  expect(container.textContent).not.toMatch(claims);
  expect(container.textContent).not.toMatch(/voyager/i);
});

test("is static", () => {
  const fetchStub = vi.fn();
  vi.stubGlobal("fetch", fetchStub);
  render(<HomePage />);
  expect(fetchStub).not.toHaveBeenCalled();
  expect(HomePage.length).toBe(0);
});

test("uses no images or external resources", () => {
  const { container } = render(<HomePage />);
  expect(container.querySelectorAll("img, svg, picture, video, iframe, [role='img']")).toHaveLength(0);
  for (const element of container.querySelectorAll("*")) {
    for (const attribute of element.attributes) {
      expect(attribute.value.startsWith("http"), `${element.tagName} ${attribute.name}`).toBe(false);
    }
  }
});

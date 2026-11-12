import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import QueryForm from "./QueryForm.jsx";
import contract from "../fixtures/contract.json";

const liveHealth = {
  ...contract.health,
  mode: "live",
  capabilities: {
    structure: { available: true, reason: null, detail: "the Voyager language model" },
    embed: { available: true, reason: null, detail: "the embedding model" },
  },
};
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const RESUME = "resume_to_posting";
const POSTING = "posting_to_resume";

const examplesFetch = () =>
  vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => contract.examples });

function setup({ direction = RESUME, health = liveHealth, busy = false } = {}) {
  const onSubmit = vi.fn();
  const fetchImpl = examplesFetch();
  const user = userEvent.setup();
  render(<QueryForm direction={direction} health={health} busy={busy} onSubmit={onSubmit} fetchImpl={fetchImpl} />);
  return { onSubmit, fetchImpl, user };
}

const choose = (user, name) => user.click(screen.getByRole("button", { name }));
test("builds requests for each method", async () => {
  const { onSubmit, fetchImpl, user } = setup();
  await user.type(screen.getByLabelText(/Resume text/), "I write Python");
  await user.selectOptions(screen.getByLabelText("Number of results"), "20");
  await user.selectOptions(screen.getByLabelText("Ranking method"), "B2");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  expect(onSubmit).toHaveBeenLastCalledWith({
    direction: RESUME, k: 20, system: "B2", query: { kind: "text", text: "I write Python" },
  });

  await choose(user, "Fill in a form");
  await user.type(screen.getByLabelText("Job title"), "Data Analyst");
  await user.type(screen.getByLabelText("Years of experience"), "3");
  await user.type(screen.getByLabelText("Skills"), "SQL, Excel\nsql");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  expect(onSubmit).toHaveBeenLastCalledWith({
    direction: RESUME, k: 20, system: "B2",
    query: { kind: "form", title: "Data Analyst", years_experience: 3, skills: ["SQL", "Excel"] },
  });

  await choose(user, "Pick an example");
  await user.selectOptions(await screen.findByRole("combobox", { name: /Example/ }), "r-2");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  expect(fetchImpl.mock.calls[0][0]).toBe("/api/examples?side=resume&limit=6");
  expect(onSubmit).toHaveBeenLastCalledWith({
    direction: RESUME, k: 20, system: "B2", query: { kind: "corpus", id: "r-2" },
  });
});

test("builds posting requests with title and level", async () => {
  const { onSubmit, user } = setup({ direction: POSTING });
  await user.type(screen.getByLabelText("Job title (optional)"), "Engineer");
  await user.type(screen.getByLabelText(/Job posting text/), "Needs React");
  await user.click(screen.getByRole("button", { name: "Find candidates" }));
  expect(onSubmit).toHaveBeenLastCalledWith({
    direction: POSTING, k: 10, system: "Full", query: { kind: "text", text: "Needs React", title: "Engineer" },
  });
  await choose(user, "Fill in a form");
  await user.selectOptions(screen.getByLabelText("Experience level"), "Associate");
  await user.type(screen.getByLabelText("Skills"), "React");
  await user.click(screen.getByRole("button", { name: "Find candidates" }));
  expect(onSubmit).toHaveBeenLastCalledWith({
    direction: POSTING, k: 10, system: "Full",
    query: { kind: "form", title: "", level: "Associate", skills: ["React"] },
  });
});

test("validates before submitting", async () => {
  const { onSubmit, user } = setup();
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  expect(screen.getByRole("alert")).toHaveTextContent("Paste some text first.");
  await choose(user, "Fill in a form");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  expect(screen.getByRole("alert")).toHaveTextContent(/title or at least one skill/);
  await choose(user, "Pick an example");
  await screen.findByRole("combobox", { name: /Example/ });
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  expect(screen.getByRole("alert")).toHaveTextContent("Choose an example first.");
  expect(onSubmit).not.toHaveBeenCalled();
});

test("resume and posting fields", async () => {
  const first = setup();
  await choose(first.user, "Fill in a form");
  expect(screen.getByLabelText("Years of experience")).toHaveAttribute("type", "number");
  expect(screen.queryByLabelText("Experience level")).not.toBeInTheDocument();
  cleanup();
  const second = setup({ direction: POSTING });
  await choose(second.user, "Fill in a form");
  const options = screen.getAllByRole("option").map((o) => o.textContent);
  for (const level of liveHealth.levels.filter((l) => l !== "Not specified")) expect(options).toContain(level);
  expect(screen.queryByLabelText("Years of experience")).not.toBeInTheDocument();
});

test("disables unavailable methods with a reason", () => {
  setup({ health: contract.health });
  expect(screen.getByRole("button", { name: "Paste text" })).toBeDisabled();
  expect(screen.getByText("Demo mode has no language model.")).toBeVisible();
  expect(screen.getByRole("button", { name: "Fill in a form" })).toBeEnabled();
  expect(screen.getByRole("button", { name: "Fill in a form" })).toHaveAttribute("aria-pressed", "true");
});

test("labels and privacy line", async () => {
  const { user } = setup();
  expect(screen.getByText("Text you paste is sent to the Voyager language model to be read.")).toBeInTheDocument();
  const area = screen.getByLabelText(/Resume text/);
  expect(area).toHaveAccessibleName(/Resume text/);
  expect(screen.getByText("required", { selector: ".sr-only" })).toBeInTheDocument();
  expect(document.querySelector(".required-dot")).toHaveAttribute("aria-hidden", "true");
  expect(screen.getByLabelText("Load a .txt or .md file")).toBeInTheDocument();
  for (const control of document.querySelectorAll("input, select, textarea")) {
    expect(control.labels.length).toBeGreaterThan(0);
  }
  await choose(user, "Fill in a form");
  expect(screen.getByText("Add a title, skills, or both")).toBeInTheDocument();
  expect(document.querySelector(".required-dot")).toBeNull();
  await choose(user, "Pick an example");
  expect(await screen.findByRole("combobox", { name: /Example resume/ })).toBeInTheDocument();
  expect(document.querySelector(".required-dot")).not.toBeNull();
});

test("loads a text file", async () => {
  const { user } = setup();
  const file = new File(["Jane Doe, Python"], "cv.txt", { type: "text/plain" });
  await user.upload(screen.getByLabelText("Load a .txt or .md file"), file);
  await waitFor(() => expect(screen.getByLabelText(/Resume text/)).toHaveValue("Jane Doe, Python"));
});

test("disables submit while busy", () => {
  setup({ busy: true });
  expect(screen.getByRole("button", { name: "Find jobs" })).toBeDisabled();
});

test("loads examples once when no fetchImpl prop is given", async () => {
  const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => contract.examples });
  vi.stubGlobal("fetch", fetchMock);
  const user = userEvent.setup();
  render(<QueryForm direction={RESUME} health={liveHealth} onSubmit={() => {}} />);
  await user.click(screen.getByRole("button", { name: "Pick an example" }));
  await screen.findByRole("combobox", { name: /Example/ });
  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(fetchMock).toHaveBeenCalledTimes(1);
});

test("shows an inline error when a file cannot be read", async () => {
  const user = userEvent.setup();
  vi.spyOn(FileReader.prototype, "readAsText").mockImplementation(function fail() {
    this.onerror?.();
  });
  render(<QueryForm direction={RESUME} health={liveHealth} onSubmit={() => {}} fetchImpl={examplesFetch()} />);
  await user.upload(screen.getByLabelText("Load a .txt or .md file"), new File(["x"], "a.txt", { type: "text/plain" }));
  expect(await screen.findByText("That file could not be read.")).toBeInTheDocument();
});

test("privacy line sits directly under the textarea", () => {
  setup();
  const area = screen.getByLabelText(/Resume text/);
  const line = screen.getByText(/Text you paste is sent to/);
  expect(area.parentElement.nextElementSibling).toBe(line);
});

test("example picker is a labeled select", async () => {
  const { user } = setup();
  await choose(user, "Pick an example");
  const select = await screen.findByRole("combobox", { name: /Example resume/ });
  expect(select).toHaveAccessibleName(/required/);
  expect(screen.queryAllByRole("radio")).toHaveLength(0);
  const options = within(select).getAllByRole("option");
  expect(options[0]).toHaveTextContent("Choose an example");
  expect(options[0]).toHaveValue("");
  expect(options).toHaveLength(contract.examples.items.length + 1);
  const item = contract.examples.items[0];
  expect(options[1]).toHaveTextContent(`${item.title || "Untitled"}, ${item.level}, ${item.skills.length} skills`);
  expect(options[1]).toHaveValue(item.id);
});

test("example picker is labeled for the posting side", async () => {
  const { user } = setup({ direction: POSTING });
  await choose(user, "Pick an example");
  expect(await screen.findByRole("combobox", { name: /Example posting/ })).toBeInTheDocument();
});

test("choosing an example sends a corpus query", async () => {
  const { onSubmit, user } = setup();
  await choose(user, "Pick an example");
  await user.selectOptions(await screen.findByRole("combobox", { name: /Example resume/ }), "r-2");
  await user.selectOptions(screen.getByLabelText("Number of results"), "5");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  expect(onSubmit).toHaveBeenCalledWith({ direction: RESUME, k: 5, system: "Full", query: { kind: "corpus", id: "r-2" } });
});

test("example loading, error and empty choice", async () => {
  const user = userEvent.setup();
  const onSubmit = vi.fn();
  const pending = vi.fn(() => new Promise(() => {}));
  const view = render(<QueryForm direction={RESUME} health={liveHealth} onSubmit={onSubmit} fetchImpl={pending} />);
  await choose(user, "Pick an example");
  expect(screen.getByRole("status")).toHaveTextContent("Loading examples…");
  expect(screen.queryByRole("combobox", { name: /Example/ })).not.toBeInTheDocument();
  view.unmount();

  const failing = vi.fn().mockRejectedValue(new TypeError("down"));
  const second = render(<QueryForm direction={RESUME} health={liveHealth} onSubmit={onSubmit} fetchImpl={failing} />);
  await choose(user, "Pick an example");
  expect(await screen.findByRole("alert")).toHaveTextContent("Cannot reach the GetWork server");
  second.unmount();

  render(<QueryForm direction={RESUME} health={liveHealth} onSubmit={onSubmit} fetchImpl={examplesFetch()} />);
  await choose(user, "Pick an example");
  const select = await screen.findByRole("combobox", { name: /Example resume/ });
  await user.selectOptions(select, "r-2");
  await user.selectOptions(select, "");
  expect(select).toHaveValue("");
  await user.click(screen.getByRole("button", { name: "Find jobs" }));
  expect(screen.getByRole("alert")).toHaveTextContent("Choose an example first.");
  expect(onSubmit).not.toHaveBeenCalled();
});

test("the three controls share one row", async () => {
  const { user } = setup();
  await choose(user, "Pick an example");
  await screen.findByRole("combobox", { name: /Example resume/ });
  const rows = document.querySelectorAll("div.control-row");
  expect(rows).toHaveLength(1);
  const cells = [...rows[0].children];
  expect(cells).toHaveLength(3);
  for (const cell of cells) {
    expect(cell).toHaveClass("field");
    expect(cell.querySelectorAll("select")).toHaveLength(1);
    expect(cell.querySelector("select").labels).toHaveLength(1);
    expect(cell.querySelector("select")).not.toHaveAttribute("class");
  }
  expect(within(cells[0]).getByRole("combobox")).toHaveAccessibleName(/Example resume/);
  expect(cells[1]).toHaveTextContent("Number of results");
  expect(cells[2]).toHaveTextContent("Ranking method");
});

test("the row holds two controls without the example method", async () => {
  const { user } = setup();
  for (const name of ["Paste text", "Fill in a form"]) {
    await choose(user, name);
    const cells = [...document.querySelector("div.control-row").children];
    expect(cells.map((c) => c.querySelector("label").textContent)).toEqual(["Number of results", "Ranking method"]);
  }
});

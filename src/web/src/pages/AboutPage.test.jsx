import { render, screen, within } from "@testing-library/react";
import { expect, test } from "vitest";
import AboutPage from "./AboutPage.jsx";
import { DISCLAIMER } from "../Layout.jsx";
import contract from "../fixtures/contract.json";

const live = {
  ...contract.health,
  mode: "artifacts",
  versions: {
    ...contract.health.versions,
    demo: false,
  },
  capabilities: {
    structure: { available: true, reason: null, detail: "the Voyager service" },
    embed: { available: true, reason: null, detail: "local" },
  },
};

test("shows methods, clusters and no data section", () => {
  render(<AboutPage health={live} />);
  expect(screen.getByRole("heading", { level: 1, name: "About GetWork" })).toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "Data in use" })).not.toBeInTheDocument();
  expect(screen.queryByText("Artifacts directory")).not.toBeInTheDocument();
  for (const term of ["GetWork (skill clusters) (Full)", "Embeddings only (B2)", "Keywords only (B1+)"]) {
    expect(screen.getByRole("heading", { level: 3, name: term })).toBeInTheDocument();
  }
  expect(screen.getAllByText("Partial credit")).toHaveLength(3);
  expect(screen.getAllByText("This is the default.")).toHaveLength(1);
  expect(screen.getByRole("heading", { level: 2, name: "Skill clusters" })).toBeInTheDocument();
  expect(screen.getByText(/JS and JavaScript count as one skill/)).toBeInTheDocument();
  expect(screen.getByText(/Only skills that fit no cluster go into one group called Other/)).toBeInTheDocument();
  expect(screen.getByText(/sent to the Voyager service to be read/)).toBeInTheDocument();
  expect(screen.getByText(/Omar Abuasba and Prathyush Vasa/)).toBeInTheDocument();
  expect(screen.getByText(DISCLAIMER)).toBeInTheDocument();
  expect(screen.getByText(/rule-based, no language model/)).toBeInTheDocument();
});

test("capability reasons", () => {
  render(<AboutPage health={contract.health} />);
  expect(screen.queryByText(/built-in demo data/)).not.toBeInTheDocument();
  expect(screen.getByText(/Reading pasted text: not available. Demo mode has no language model./)).toBeInTheDocument();
  expect(screen.getByText(/Embeddings: available/)).toBeInTheDocument();
  expect(screen.getByText(/Pasted text cannot be read on this server\./)).toBeInTheDocument();
  expect(screen.queryByText(/sent to/)).not.toBeInTheDocument();
});

const card = (label) => within(screen.getByRole("heading", { level: 3, name: label }).closest("li"));

test("each method card holds its own facts", () => {
  render(<AboutPage health={live} />);
  const full = card("GetWork (skill clusters) (Full)");
  expect(full.getByText("This is the default.")).toBeInTheDocument();
  expect(full.getByText(/documents that share a cluster name are meant to move closer/)).toBeInTheDocument();
  expect(full.getByText(/React and Vue under Front-end count as related/)).toBeInTheDocument();
  expect(full.getByText(/Skills under Other still count for exact matches/)).toBeInTheDocument();
  expect(full.getByText(/This is not a scoring rule/)).toBeInTheDocument();
  const b2 = card("Embeddings only (B2)");
  expect(b2.queryByText("This is the default.")).not.toBeInTheDocument();
  expect(b2.getByText(/Same text as Full but without the cluster names/)).toBeInTheDocument();
  expect(b2.getByText(/Compared with Full \(same text minus the cluster names\), it shows the effect of the cluster names/)).toBeInTheDocument();
  expect(b2.getByText(/it gets no cluster labels.*skill order can still carry a little cluster information/)).toBeInTheDocument();
  expect(b2.getByText(/no explicit signal that two skills belong together/)).toBeInTheDocument();
  const b1 = card("Keywords only (B1+)");
  expect(b1.queryByText("This is the default.")).not.toBeInTheDocument();
  expect(b1.getByText(/closer levels score higher/)).toBeInTheDocument();
  expect(b1.getByText(/No credit for related skills/)).toBeInTheDocument();
  expect(b1.getByText(/By default, skills carry the most weight \(70%\), title words and level 15% each/)).toBeInTheDocument();
  expect(b1.getByText(/normalising spelling/)).toBeInTheDocument();
});

test("clusters are named and reviewed", () => {
  render(<AboutPage health={live} />);
  expect(screen.getByText(/named, then reviewed \(a model flags names that do not fit, and the team reviews them\)/)).toBeInTheDocument();
});

test("the cluster example names React and Vue", () => {
  render(<AboutPage health={live} />);
  const example = screen.getByRole("heading", { level: 3, name: "Example" });
  expect(example.nextElementSibling).toHaveTextContent(/React.*Vue.*Front-end/);
  expect(example.nextElementSibling).toHaveTextContent(/^For example/);
});

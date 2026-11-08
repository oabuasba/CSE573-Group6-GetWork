import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import AboutPage from "./AboutPage.jsx";
import { DISCLAIMER } from "../Layout.jsx";
import contract from "../fixtures/contract.json";

const live = {
  ...contract.health,
  mode: "artifacts",
  artifacts_dir: "artifacts_trial",
  versions: {
    ...contract.health.versions,
    demo: false,
    embedding: { model: "embed-x", backend: "local", dim: 384 },
  },
  capabilities: {
    structure: { available: true, reason: null, detail: "the Voyager service" },
    embed: { available: true, reason: null, detail: "local" },
  },
};

test("shows health data and methods", () => {
  render(<AboutPage health={live} />);
  expect(screen.getByRole("heading", { level: 1, name: "About GetWork" })).toBeInTheDocument();
  expect(screen.getByText("artifacts_trial")).toBeInTheDocument();
  expect(screen.getByText("embed-x (local, 384 dimensions)")).toBeInTheDocument();
  expect(screen.getByText("Resumes").nextElementSibling).toHaveTextContent("6");
  for (const term of ["GetWork (skill clusters) (Full)", "Embeddings only (B2)", "Keywords only (B1+)"]) {
    expect(screen.getByText(term)).toBeInTheDocument();
  }
  expect(screen.getByText(/sent to the Voyager service to be read/)).toBeInTheDocument();
  expect(screen.getByText(/Omar Abuasba and Prathyush Vasa/)).toBeInTheDocument();
  expect(screen.getByText(DISCLAIMER)).toBeInTheDocument();
  expect(screen.getByText(/rule-based, no language model/)).toBeInTheDocument();
  expect(screen.queryByText("This server is running on built-in demo data.")).not.toBeInTheDocument();
});

test("capability reasons and demo", () => {
  render(<AboutPage health={contract.health} />);
  expect(screen.getByText("This server is running on built-in demo data.")).toBeInTheDocument();
  expect(screen.getByText(/Reading pasted text: not available. Demo mode has no language model./)).toBeInTheDocument();
  expect(screen.getByText(/Embeddings: available/)).toBeInTheDocument();
  expect(screen.getByText(/Pasted text cannot be read on this server\./)).toBeInTheDocument();
  expect(screen.queryByText(/sent to/)).not.toBeInTheDocument();
});

import { DISCLAIMER } from "../Layout.jsx";
import { SYSTEM_LABELS } from "../api.js";

const METHOD_TEXT = {
  Full: "Compares documents through shared skill clusters, so related skills earn partial credit.",
  B2: "Compares one embedding of each whole document.",
  "B1+": "Scores matching skill words, with no embedding model.",
};
const CAPABILITY_NAMES = { structure: "Reading pasted text", embed: "Embeddings" };

const show = (value) => (value == null ? "none" : String(value));

function describeEmbedding(embedding) {
  return embedding ? `${embedding.model} (${embedding.backend}, ${embedding.dim} dimensions)` : "none";
}

function Row({ term, children }) {
  return (
    <>
      <dt>{term}</dt>
      <dd>{children}</dd>
    </>
  );
}

function DataInUse({ health }) {
  const { versions, counts } = health;
  return (
    <section>
      <h2>Data in use</h2>
      {versions.demo && <p>This server is running on built-in demo data.</p>}
      <dl>
        <Row term="Mode">{health.mode}</Row>
        <Row term="Resumes">{counts.resumes}</Row>
        <Row term="Postings">{counts.postings}</Row>
        <Row term="Artifacts directory">{show(health.artifacts_dir)}</Row>
        <Row term="Alias table version">{show(versions.alias_table_version)}</Row>
        <Row term="Taxonomy version">{show(versions.taxonomy_version)}</Row>
        <Row term="Serialization version">{show(versions.serialization_version)}</Row>
        <Row term="Relabel mode">{show(versions.relabel_mode)}</Row>
        <Row term="Embedding model">{describeEmbedding(versions.embedding)}</Row>
      </dl>
    </section>
  );
}

function Capabilities({ health }) {
  const { capabilities, explainer } = health;
  return (
    <section>
      <h2>Server capabilities</h2>
      <ul>
        {Object.entries(CAPABILITY_NAMES).map(([key, name]) => {
          const { available, reason } = capabilities[key];
          return (
            <li key={key}>
              {name}: {available ? "available" : `not available. ${reason ?? ""}`}
            </li>
          );
        })}
      </ul>
      <p>
        Explanations: {explainer.kind === "template" ? "rule-based, no language model" : explainer.model}.
      </p>
    </section>
  );
}

export default function AboutPage({ health }) {
  const { detail } = health.capabilities.structure;
  return (
    <>
      <h1>About GetWork</h1>
      <section className="measure">
        <h2>What it does</h2>
        <p>
          GetWork matches resumes to job postings, and postings to resumes, using skill clusters.
          Each explanation lists the skills two documents share, the related skills, and the missing ones.
        </p>
      </section>
      <section>
        <h2>Ranking methods</h2>
        <dl>
          {Object.entries(SYSTEM_LABELS).map(([id, label]) => (
            <Row key={id} term={`${label} (${id})`}>{METHOD_TEXT[id]}</Row>
          ))}
        </dl>
      </section>
      <DataInUse health={health} />
      <Capabilities health={health} />
      <section className="measure">
        <h2>Privacy</h2>
        <p>
          The server keeps no history and writes nothing.{" "}
          {detail ? `Text you paste is sent to ${detail} to be read.` : "Pasted text cannot be read on this server."}
        </p>
      </section>
      <section>
        <h2>The team</h2>
        <p>Omar Abuasba and Prathyush Vasa, CSE 573 Group 6.</p>
      </section>
      <p className="measure">{DISCLAIMER}</p>
    </>
  );
}

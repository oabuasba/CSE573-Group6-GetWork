import { DISCLAIMER } from "../Layout.jsx";
import { SYSTEM_LABELS } from "../api.js";

const DEFAULT_METHOD = "Full";
const METHODS = {
  Full: {
    compares: "Each document as text: title, level and skills grouped under cluster names, read as meaning by an embedding model.",
    credit: "By design, documents that share a cluster name are meant to move closer in the embedding, so related skills are meant to earn partial credit. This is not a scoring rule.",
    strengths: "Uses the skill clusters, so for example React and Vue under Front-end count as related.",
    limits: "Skills under Other still count for exact matches but earn no partial credit.",
  },
  B2: {
    compares: "Same text as Full but without the cluster names, read as meaning by the same kind of embedding model.",
    credit: "Not by design: it gets no cluster labels, so relatedness comes from what the embedding model already knows (the skill order can still carry a little cluster information).",
    strengths: "Compared with Keywords only, it shows the effect of the embeddings. Compared with Full (same text minus the cluster names), it shows the effect of the cluster names.",
    limits: "Without cluster names, the documents carry no explicit signal that two skills belong together.",
  },
  "B1+": {
    compares: "The skills the two documents list, the words in their titles, and their experience levels. No embedding model.",
    credit: "No credit for related skills. A skill counts only when both documents list it (compared after normalising spelling). For experience level, closer levels score higher.",
    strengths: "By default, skills carry the most weight (70%), title words and level 15% each.",
    limits: "Misses related skills and different wording for the same job.",
  },
};
const METHOD_ROWS = [
  ["compares", "What it compares"],
  ["credit", "Partial credit"],
  ["strengths", "Strengths and notes"],
  ["limits", "Limits"],
];
const CAPABILITY_NAMES = { structure: "Reading pasted text", embed: "Embeddings" };

function Row({ term, children }) {
  return (
    <>
      <dt>{term}</dt>
      <dd>{children}</dd>
    </>
  );
}

function MethodCard({ id }) {
  const method = METHODS[id];
  return (
    <li className="card">
      <h3>{`${SYSTEM_LABELS[id]} (${id})`}</h3>
      {id === DEFAULT_METHOD && <p>This is the default.</p>}
      <dl>
        {METHOD_ROWS.map(([key, term]) => (
          <Row key={key} term={term}>{method[key]}</Row>
        ))}
      </dl>
    </li>
  );
}

function RankingMethods() {
  return (
    <section>
      <h2>Ranking methods</h2>
      <p className="measure">
        Every method scores each resume and posting pair, then lists the highest scores first.
        Use the method menu on the search page to switch between them.
      </p>
      <ul className="result-list">
        {Object.keys(SYSTEM_LABELS).map((id) => (
          <MethodCard key={id} id={id} />
        ))}
      </ul>
    </section>
  );
}

function SkillClusters() {
  return (
    <section className="measure">
      <h2>Skill clusters</h2>
      <p>
        A skill cluster is a named group of skills with similar meaning. For example, Front-end or Cloud (real cluster names depend on the run).
      </p>
      <h3>How clusters are made</h3>
      <ol>
        <li>Spellings of the same skill are merged first, so JS and JavaScript count as one skill.</li>
        <li>An embedding model turns each common skill from the resumes and postings into numbers that capture its meaning.</li>
        <li>Skills with similar meaning are grouped automatically.</li>
        <li>Each group is named, then reviewed (a model flags names that do not fit, and the team reviews them).</li>
        <li>Only skills that fit no cluster go into one group called Other. Skills under Other get no partial credit.</li>
      </ol>
      <h3>How clusters are used</h3>
      <ul>
        <li>In the Full method, each document lists its skills under their cluster names, so related skills earn partial credit.</li>
        <li>Results show skills grouped by cluster, with Other last.</li>
        <li>Explanations list the skills two documents share, the related skills, and the missing ones.</li>
      </ul>
      <h3>Example</h3>
      <p>
        For example, a resume lists React and a posting asks for Vue. If both sit under Front-end, then the Full method treats them as
        related and gives partial credit. Real membership depends on the data. Keywords only sees two different words and gives no skill credit.
      </p>
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
          GetWork matches resumes to job postings, and postings to resumes, by comparing skills, titles and
          experience levels. You choose how the comparison is scored.
        </p>
      </section>
      <RankingMethods />
      <SkillClusters />
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

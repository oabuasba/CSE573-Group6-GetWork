import "../theme/home.css";
import { hrefFor } from "../routes.js";

const STEPS = [
  {
    title: "Share a resume or a posting",
    text: "Paste the text, fill in a short form, or pick an example. Use Find jobs when you have a resume, and Find candidates when you have a posting.",
  },
  {
    title: "GetWork reads it",
    text: "A language model reads pasted text and pulls out the title, skills and experience. Spellings of the same skill are merged, so JS and JavaScript count as one, and each skill is placed in a skill cluster, a named group of similar skills such as Front-end. Which input options are available depends on how the server was set up, and the page tells you when one is not.",
  },
  {
    title: "It finds the closest matches",
    text: "Each document becomes a list of numbers, an embedding, that captures its meaning and is built from the title, level and skills grouped under their cluster names. GetWork compares your document with every document on the other side and lists the highest scores first.",
  },
  {
    title: "You see why, and can ask more",
    text: "Every match lists the skills you share, the related skills and the missing ones, with a short summary written from those lists by rules. Where the chat agent is switched on, you can open a chat about one suggestion; its answers are checked against the two documents before they are shown.",
  },
];

const BENEFITS = [
  {
    title: "Both directions",
    text: "Job seekers get ranked postings for a resume. Recruiters and hiring managers get ranked candidates for a posting.",
  },
  {
    title: "Related skills earn partial credit",
    text: "A resume that lists Vue can still earn credit against a posting that asks for React when both skills sit in the same cluster, such as Front-end. Which skills share a cluster depends on the data.",
  },
  {
    title: "Reasons you can read",
    text: "Shared, related and missing skills are listed for every match, with a summary in plain language, so you can see what lines up and what is absent.",
  },
  {
    title: "Choose how matches are scored",
    text: "Switch between the skill-cluster method, embeddings only and keywords only to see what each one adds. The ",
    aboutLink: true,
    after: " explains all three.",
  },
  {
    title: "Ask follow-up questions",
    text: "Where the chat agent is switched on, ask about one suggestion, for example which skills are missing. Claims that cannot be checked against the two documents are left out.",
  },
  {
    title: "Nothing kept on the server",
    text: "The server keeps no history and writes nothing. Pasted text goes to the language model only so it can be read, and your results and chats stay in this browser tab until you close it.",
  },
];

function Benefit({ title, text, aboutLink, after }) {
  return (
    <li className="card">
      <h3>{title}</h3>
      <p>
        {text}
        {aboutLink && <a href={hrefFor("about")}>About page</a>}
        {after}
      </p>
    </li>
  );
}

export default function HomePage() {
  return (
    <>
      <section className="home-hero" aria-labelledby="home-title">
        <h1 id="home-title">Match resumes to jobs, and jobs to resumes</h1>
        <p className="home-lead">
          GetWork ranks job postings for a resume, or candidates for a posting, by comparing skills, job titles and
          experience levels. Its default method gives partial credit to related skills, such as React and Vue, and every
          match comes with the reasons behind it.
        </p>
        <div className="home-actions">
          <a className="btn btn-cta" href={hrefFor("jobs")}>Find jobs for my resume</a>
          <a className="btn btn-secondary" href={hrefFor("candidates")}>Find candidates for a posting</a>
        </div>
        <p className="home-hint">No resume or posting at hand? Choose &quot;Pick an example&quot; on the next page.</p>
      </section>

      <section className="home-section" aria-labelledby="home-how">
        <h2 id="home-how">How it works</h2>
        <p className="measure">Four steps from a document to a ranked list.</p>
        <ol className="home-steps">
          {STEPS.map((step) => (
            <li className="card home-step" key={step.title}>
              <h3>{step.title}</h3>
              <p>{step.text}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="home-section" aria-labelledby="home-benefits">
        <h2 id="home-benefits">What you get</h2>
        <ul className="home-features">
          {BENEFITS.map((benefit) => (
            <Benefit key={benefit.title} {...benefit} />
          ))}
        </ul>
      </section>

      <section className="home-section measure" aria-labelledby="home-honest">
        <h2 id="home-honest">What GetWork is, and is not</h2>
        <p>
          GetWork is a CSE 573 course project at Arizona State University, built by Group 6. It is not an official ASU
          website and it is not a job board.
        </p>
        <p>
          It matches against a fixed set of resumes and job postings collected for the project from public data sets (a
          small built-in sample in demo mode), not against live listings. How well its rankings agree with human
          judgement has not been formally evaluated yet, so treat the results as suggestions to look at, not as
          decisions.
        </p>
        <p>
          The ranking methods, skill clusters and privacy details are on the <a href={hrefFor("about")}>About page</a>.
        </p>
      </section>

      <section className="home-section card home-cta" aria-labelledby="home-cta-title">
        <h2 id="home-cta-title">Ready to see your matches?</h2>
        <p>Start with a resume or a posting.</p>
        <div className="home-actions">
          <a className="btn btn-primary" href={hrefFor("jobs")}>Try it with your resume</a>
          <a className="btn btn-secondary" href={hrefFor("candidates")}>Try it with a job posting</a>
        </div>
      </section>
    </>
  );
}

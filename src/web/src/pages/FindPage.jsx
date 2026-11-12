import { useEffect, useId, useRef, useState } from "react";
import { describeError, NETWORK_ERROR_CODE, postMatch } from "../api.js";
import ChatOnDemand from "../components/ChatOnDemand.jsx";
import ExplainPanel from "../components/ExplainPanel.jsx";
import GroupedSkills from "../components/GroupedSkills.jsx";
import MatchCard from "../components/MatchCard.jsx";
import Notice from "../components/Notice.jsx";
import QueryForm from "../components/QueryForm.jsx";

const COPY = {
  resume_to_posting: {
    heading: "Find jobs for your resume",
    intro: "Give us your resume and we rank the job postings that fit your skills best.",
    noun: "postings",
  },
  posting_to_resume: {
    heading: "Find candidates for your posting",
    intro: "Give us a job posting and we rank the resumes that fit it best.",
    noun: "candidates",
  },
};
const IDLE = { status: "idle", response: null, error: null, system: null };

function UnderstoodCard({ query }) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  return (
    <section className="card" aria-labelledby="understood-heading">
      <h3 id="understood-heading">What we understood</h3>
      <button
        type="button"
        className="btn btn-secondary"
        aria-expanded={open}
        aria-controls={panelId}
        aria-describedby="understood-heading"
        onClick={() => setOpen(!open)}
      >
        {open ? "Hide details" : "Show details"}
      </button>
      <div id={panelId} hidden={!open}>
        {open && (
          <>
            <p>
              {query.title || "Untitled"}, {query.level}
              {query.years_experience != null && `, ${query.years_experience} years of experience`}
            </p>
            <GroupedSkills skills={query.skills} />
          </>
        )}
      </div>
    </section>
  );
}

function ErrorAlert({ error }) {
  const { title, body, hint } = describeError(error);
  const isNetwork = error.code === NETWORK_ERROR_CODE;
  return (
    <Notice kind="error" role="alert">
      <strong>{title}.</strong> {!isNetwork && body} {hint}
      {isNetwork && (
        <>
          {" "}
          Run <code>uv run webapp serve</code>.
        </>
      )}
    </Notice>
  );
}

function Results({ response, copy, headingRef, expandedId, onToggle, fetchImpl, chatEnabled }) {
  const { matches, notes, query, system, direction } = response;
  const chatMatches = matches.map((m) => ({ id: m.id, rank: m.rank, score: m.score }));
  return (
    <section>
      <h2 tabIndex={-1} ref={headingRef}>Results</h2>
      <p aria-live="polite">{matches.length} {copy.noun} found</p>
      <UnderstoodCard query={query} />
      {notes.map((note) => (
        <Notice key={note.code} kind="warning" label="Note">{note.message}</Notice>
      ))}
      {matches.length === 0 ? (
        <div className="card">
          <h3>No matches</h3>
          <p>Try adding more skills, or choose another ranking method.</p>
        </div>
      ) : (
        <ol className="result-list">
          {matches.map((match) => (
            <li key={match.id}>
              <MatchCard
                match={match}
                system={system}
                expanded={expandedId === match.id}
                onToggle={() => onToggle(match.id)}
              >
                <GroupedSkills skills={match.document.skills} />
                <ExplainPanel direction={direction} query={query} matchId={match.id} fetchImpl={fetchImpl} />
                {chatEnabled && (
                  <ChatOnDemand
                    direction={direction}
                    query={query}
                    focusId={match.id}
                    matches={chatMatches}
                    system={system}
                    fetchImpl={fetchImpl}
                  />
                )}
              </MatchCard>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

export default function FindPage({ direction, health, fetchImpl }) {
  const copy = COPY[direction];
  const [state, setState] = useState(IDLE);
  const [expandedId, setExpandedId] = useState(null);
  const [lastKind, setLastKind] = useState(null);
  const headingRef = useRef(null);

  useEffect(() => {
    if (state.status === "done") headingRef.current?.focus();
  }, [state]);

  async function submit(request) {
    setState({ ...IDLE, status: "loading" });
    setLastKind(request.query.kind);
    setExpandedId(null);
    try {
      const response = await postMatch(request, fetchImpl);
      setState({ status: "done", response, error: null });
    } catch (error) {
      setState({ status: "error", response: null, error });
    }
  }

  const busy = state.status === "loading";
  return (
    <>
      <h1>{copy.heading}</h1>
      <p className="measure">{copy.intro}</p>
      <QueryForm key={direction} direction={direction} health={health} busy={busy}
        onSubmit={submit} fetchImpl={fetchImpl} />
      {busy && (
        <p role="status">
          Finding matches...
          {lastKind === "text" && " Reading your document with the language model can take up to a minute."}
        </p>
      )}
      {state.status === "error" && <ErrorAlert error={state.error} />}
      {state.status === "done" && (
        <Results
          response={state.response} copy={copy} headingRef={headingRef} fetchImpl={fetchImpl}
          chatEnabled={health?.explainer?.kind === "agent"}
          expandedId={expandedId} onToggle={(id) => setExpandedId(expandedId === id ? null : id)}
        />
      )}
    </>
  );
}

import { useCallback, useEffect, useId, useRef, useState } from "react";
import { describeError, postExplain } from "../api.js";
import ClaimList from "./ClaimList.jsx";
import Notice from "./Notice.jsx";

function explainerLine(explainer) {
  return explainer.kind === "template"
    ? "Explained by rules, no language model"
    : `Explained by ${explainer.model}`;
}

function AgentSummary({ claims, facts }) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  if (claims.length === 1) return <ClaimList claims={claims} facts={facts} />;
  return (
    <>
      {!open && <p className="measure">{claims[0].text}</p>}
      <button
        type="button"
        className="btn btn-secondary"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen(!open)}
      >
        {open ? "Show less" : `Show all ${claims.length} points`}
      </button>
      <div id={panelId} hidden={!open}>
        {open && <ClaimList claims={claims} facts={facts} />}
      </div>
    </>
  );
}

export default function ExplainPanel({ direction, query, matchId, fetchImpl }) {
  const [state, setState] = useState({ status: "loading", data: null, error: null });
  const started = useRef(false);

  const load = useCallback(() => {
    setState({ status: "loading", data: null, error: null });
    postExplain({ direction, query, match_id: matchId }, fetchImpl).then(
      (data) => setState({ status: "ready", data, error: null }),
      (error) => setState({ status: "error", data: null, error }),
    );
  }, [direction, query, matchId, fetchImpl]);

  useEffect(() => {
    // StrictMode runs effects twice in development; the ref keeps it to one request.
    if (started.current) return;
    started.current = true;
    load();
  }, [load]);

  if (state.status === "loading") return <p role="status">Loading the explanation...</p>;
  if (state.status === "error") {
    const { title, body, hint } = describeError(state.error);
    return (
      <Notice kind="error" role="alert">
        <strong>{title}.</strong> {body} {hint}{" "}
        <button type="button" className="btn btn-secondary" onClick={load}>
          Try again
        </button>
      </Notice>
    );
  }
  const { summary, bullets, explainer, claims, facts } = state.data;
  return (
    <div>
      {claims?.length ? (
        <AgentSummary claims={claims} facts={facts ?? []} />
      ) : (
        <>
          <p className="measure">{summary}</p>
          <ul>
            {bullets.map((bullet) => (
              <li key={bullet}>{bullet}</li>
            ))}
          </ul>
        </>
      )}
      <p>{explainerLine(explainer)}</p>
    </div>
  );
}

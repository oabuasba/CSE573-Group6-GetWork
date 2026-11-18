import { useId } from "react";
import { CONTACT_UNAVAILABLE, DEMO_NAME_LABEL, UNNAMED_CANDIDATE } from "../candidateCopy.js";
import EvidenceChips from "./EvidenceChips.jsx";

function CardHead({ match }) {
  return (
    <>
      <p>Rank {match.rank}</p>
      <h3>{match.document.title || "Untitled"}</h3>
      <p>{match.document.level}</p>
    </>
  );
}

function CandidateHead({ match }) {
  const hasName = typeof match.display_name === "string" && match.display_name.trim() !== "";
  return (
    <>
      <p>Rank {match.rank}</p>
      <h3>{hasName ? match.display_name : UNNAMED_CANDIDATE}</h3>
      {hasName && (
        <p>
          <span className="chip chip-other" data-demo-name>{DEMO_NAME_LABEL}</span>
        </p>
      )}
      <p>{match.document.title || "Untitled"}</p>
      <p>{match.document.level}</p>
      <p>{CONTACT_UNAVAILABLE}</p>
    </>
  );
}

export default function MatchCard({ match, system, expanded, onToggle, children }) {
  const regionId = useId();
  const scoreLabel = system === "B1+" ? "Keyword score" : "Similarity";
  const isCandidate = match.document.side === "resume";
  return (
    <article className="card" data-candidate={isCandidate ? "" : undefined}>
      {isCandidate ? <CandidateHead match={match} /> : <CardHead match={match} />}
      <p>
        <strong>{match.score.toFixed(3)}</strong> <span>{scoreLabel}</span>
      </p>
      <EvidenceChips evidence={match.evidence} />
      <button
        type="button"
        className="btn btn-secondary"
        aria-expanded={expanded}
        aria-controls={regionId}
        onClick={onToggle}
      >
        Why this match?
      </button>
      <div id={regionId} role="region" aria-label="Why this match" hidden={!expanded}>
        {expanded && children}
      </div>
    </article>
  );
}

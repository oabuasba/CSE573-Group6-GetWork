import { useId } from "react";
import EvidenceChips from "./EvidenceChips.jsx";

export default function MatchCard({ match, system, expanded, onToggle, children }) {
  const regionId = useId();
  const scoreLabel = system === "B1+" ? "Keyword score" : "Similarity";
  return (
    <article className="card">
      <p>Rank {match.rank}</p>
      <h3>{match.document.title || "Untitled"}</h3>
      <p>{match.document.level}</p>
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

import { useId, useState } from "react";

function Claim({ claim, factsById, index, idPrefix }) {
  const [open, setOpen] = useState(false);
  const panelId = `${idPrefix}-claim-evidence-${index}-${claim.facts.join("-")}`;
  const cited = claim.facts.map((id) => factsById.get(id)).filter(Boolean);
  return (
    <li>
      <p className="measure">{claim.text}</p>
      <button
        type="button"
        className="btn btn-secondary"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen(!open)}
      >
        {open ? "Hide evidence" : "Show evidence"}
      </button>
      <div id={panelId} hidden={!open}>
        {open && (
          <>
            <ul>
              {cited.map((fact) => (
                <li key={fact.id}>{fact.text}</li>
              ))}
            </ul>
            {claim.skills.length > 0 && <p>Skills: {claim.skills.join(", ")}</p>}
          </>
        )}
      </div>
    </li>
  );
}

export default function ClaimList({ claims, facts }) {
  const idPrefix = useId();
  const factsById = new Map(facts.map((fact) => [fact.id, fact]));
  return (
    <ol>
      {claims.map((claim, i) => (
        <Claim key={i} claim={claim} factsById={factsById} index={i} idPrefix={idPrefix} />
      ))}
    </ol>
  );
}

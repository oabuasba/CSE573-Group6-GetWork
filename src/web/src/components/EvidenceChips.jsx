const joinSkills = (skills) => skills.join(", ");

const ROWS = [
  {
    key: "shared",
    label: "Shared",
    symbol: "✓",
    className: "chip chip-shared",
    text: (item) => item.skill,
  },
  {
    key: "related",
    label: "Related",
    symbol: "≈",
    className: "chip chip-related",
    text: (item) => `${joinSkills(item.resume_skills)} and ${joinSkills(item.posting_skills)} in ${item.cluster}`,
  },
  {
    key: "gaps",
    label: "Missing",
    symbol: "✗",
    className: "chip chip-gap",
    text: (item) => item.skill,
  },
];

export default function EvidenceChips({ evidence }) {
  return (
    <div>
      {ROWS.filter(({ key }) => evidence[key]?.length > 0).map(({ key, label, symbol, className, text }) => (
        <div key={key}>
          <strong>{label}</strong>
          <ul className="result-list" aria-label={label}>
            {evidence[key].map((item) => (
              <li key={text(item)} className={className}>
                <span aria-hidden="true">{symbol}</span> {text(item)}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

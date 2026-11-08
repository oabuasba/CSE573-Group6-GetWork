const OTHER = "Other";

function groupByCluster(skills) {
  const groups = new Map();
  for (const { skill, cluster } of skills) {
    if (!groups.has(cluster)) groups.set(cluster, []);
    groups.get(cluster).push(skill);
  }
  const named = [...groups].filter(([cluster]) => cluster !== OTHER);
  return groups.has(OTHER) ? [...named, [OTHER, groups.get(OTHER)]] : named;
}

export default function GroupedSkills({ skills }) {
  return (
    <div>
      {groupByCluster(skills).map(([cluster, names]) => (
        <section key={cluster}>
          <h4>{cluster}</h4>
          <ul className="result-list">
            {names.map((name) => (
              <li key={name} className={cluster === OTHER ? "chip chip-other" : "chip"}>
                {name}
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}

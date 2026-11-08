const MAX_SKILLS = 100;

export function parseSkills(text) {
  const seen = new Set();
  const skills = [];
  for (const part of String(text ?? "").split(/[\n,;]/)) {
    const skill = part.trim();
    const key = skill.toLowerCase();
    if (!skill || seen.has(key)) continue;
    seen.add(key);
    skills.push(skill);
  }
  return skills.slice(0, MAX_SKILLS);
}

function embedOk(system, capabilities) {
  if (system === "B1+" || capabilities.embed.available) return { ok: true };
  return { ok: false, reason: capabilities.embed.reason };
}

export function canRun(kind, system, capabilities) {
  if (kind === "corpus") return { ok: true };
  if (kind === "text" && !capabilities.structure.available) {
    return { ok: false, reason: capabilities.structure.reason };
  }
  return embedOk(system, capabilities);
}

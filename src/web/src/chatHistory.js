export const MAX_HISTORY = 20;
export const MAX_CONTENT = 2000;
export const MAX_MESSAGE = 1000;

const contentOf = (turn) => (turn.role === "assistant" ? turn.reply : turn.content) ?? "";

export function toHistory(turns) {
  const answered = turns.filter((turn, i) => turn.role !== "user" || turns[i + 1]?.role === "assistant");
  return answered
    .map((turn) => ({ role: turn.role, content: contentOf(turn).slice(0, MAX_CONTENT) }))
    .filter((turn) => turn.content !== "")
    .slice(-MAX_HISTORY);
}

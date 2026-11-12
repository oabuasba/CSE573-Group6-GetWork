export const CONVERSATION_PREFIX = "getwork.chat.v1:";

const keyFor = (matchId) => `${CONVERSATION_PREFIX}${matchId}`;

function isTurn(turn) {
  if (turn === null || typeof turn !== "object") return false;
  if (turn.role === "user") return typeof turn.content === "string";
  return turn.role === "assistant" && typeof turn.reply === "string";
}

export function loadConversation(matchId) {
  try {
    const raw = window.sessionStorage.getItem(keyFor(matchId));
    if (raw === null) return [];
    const turns = JSON.parse(raw);
    return Array.isArray(turns) && turns.every(isTurn) ? turns : [];
  } catch {
    return [];
  }
}

export function saveConversation(matchId, turns) {
  if (turns.length === 0) return clearConversation(matchId);
  try {
    window.sessionStorage.setItem(keyFor(matchId), JSON.stringify(turns));
  } catch {
    // Storage full or disabled: the conversation stays in memory only.
  }
}

export function clearConversation(matchId) {
  try {
    window.sessionStorage.removeItem(keyFor(matchId));
  } catch {
    // Storage disabled: nothing to clear.
  }
}

export function clearAllConversations() {
  try {
    const storage = window.sessionStorage;
    for (let index = storage.length - 1; index >= 0; index -= 1) {
      const key = storage.key(index);
      if (key?.startsWith(CONVERSATION_PREFIX)) storage.removeItem(key);
    }
  } catch {
    // Storage disabled: nothing to clear.
  }
}

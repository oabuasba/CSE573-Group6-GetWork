import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import ChatPage from "./ChatPage.jsx";
import { AI_DISCLAIMER } from "../components/ChatComposer.jsx";
import { GREETING } from "../components/ChatThread.jsx";
import { CONVERSATION_PREFIX, loadConversation } from "../conversationStore.js";
import contract from "../fixtures/contract.json";
import agent from "../fixtures/contract_agent.json";
import { PANEL_KEY } from "../chatPanelPref.js";
import { saveResults } from "../resultsSession.js";

const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
const health = { ...contract.health, chat: { enabled: true, model: "demo-model" } };
const [first, second] = contract.match.matches;
const NO_SESSION =
  "Open a chat from a search result: run a search, open a result with Why this match?, then choose Chat about this suggestion.";

afterEach(() => vi.restoreAllMocks());

function seed(direction = "resume_to_posting") {
  saveResults({ direction, response: contract.match, expandedId: null });
}

function routedFetch({ chat = () => reply(agent.chat) } = {}) {
  return vi.fn(async (path) => {
    if (path.startsWith("/api/chat")) return chat();
    if (path.startsWith("/api/explain")) return reply(contract.explain);
    throw new Error(`unexpected ${path}`);
  });
}

const bodies = (fetchImpl, path) =>
  fetchImpl.mock.calls.filter(([p]) => p === path).map(([, init]) => JSON.parse(init.body));
const snapshot = () => JSON.parse(window.sessionStorage.getItem("getwork.results.v1"));

function setup({ id = first.id, fetchImpl = routedFetch(), chatHealth = health } = {}) {
  seed();
  const user = userEvent.setup();
  const view = render(<ChatPage matchId={id} health={chatHealth} fetchImpl={fetchImpl} />);
  return { user, fetchImpl, ...view };
}

const replyText = (_, el) => el?.classList.contains("chat-reply") && el.textContent === agent.chat.reply;
const box = () => screen.getByLabelText("Your question");
async function ask(user, text) {
  await user.type(box(), text);
  await user.click(screen.getByRole("button", { name: "Send" }));
}

test("shows only the focus match", () => {
  setup();
  const headings = screen.getAllByRole("heading", { level: 1 });
  expect(headings).toHaveLength(1);
  expect(headings[0]).toHaveTextContent("Chat about this suggestion");
  expect(screen.getByRole("link", { name: "Back to results" })).toHaveAttribute("href", "#/jobs");
  const section = screen.getByRole("region", { name: "The suggestion" });
  expect(within(section).getByRole("heading", { name: first.document.title })).toBeInTheDocument();
  expect(within(section).getByText(`Rank ${first.rank}`)).toBeInTheDocument();
  expect(within(section).getByText("Similarity")).toBeInTheDocument();
  expect(screen.queryByText(second.document.title)).not.toBeInTheDocument();
  expect(screen.queryByText("What we understood")).not.toBeInTheDocument();
  expect(screen.queryByText(/found$/)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Why this match?" })).toHaveAttribute("aria-expanded", "false");
  const order = [...document.querySelectorAll("a.link, h1, section, form")].map((el) => el.tagName);
  expect(order).toEqual(["A", "H1", "SECTION", "FORM", "SECTION"]);
});

test("back link follows the searched direction", () => {
  seed("posting_to_resume");
  render(<ChatPage matchId={first.id} health={health} fetchImpl={routedFetch()} />);
  expect(screen.getByRole("link", { name: "Back to results" })).toHaveAttribute("href", "#/candidates");
});

test("why this match opens on demand", async () => {
  const { user, fetchImpl } = setup();
  expect(fetchImpl).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "Why this match?" }));
  expect(await screen.findByText(contract.explain.summary)).toBeInTheDocument();
  expect(bodies(fetchImpl, "/api/explain")).toEqual([
    { direction: "resume_to_posting", query: contract.match.query, match_id: first.id },
  ]);
});

test("mounting makes no network call and marks returning", () => {
  const { fetchImpl } = setup();
  expect(fetchImpl).not.toHaveBeenCalled();
  expect(snapshot().returning).toBe(true);
  expect(box()).toHaveFocus();
});

test("send, pending, reply and focus", async () => {
  let release;
  const fetchImpl = routedFetch({ chat: () => new Promise((resolve) => (release = () => resolve(reply(agent.chat)))) });
  const { user } = setup({ fetchImpl });
  await ask(user, "Why?");
  expect(screen.getByRole("log")).toHaveTextContent("You: Why?");
  expect(screen.getByRole("status")).toHaveTextContent("The assistant is thinking…");
  expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  expect(box()).toBeEnabled();
  release();
  expect(await screen.findByText(replyText)).toBeInTheDocument();
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
  await waitFor(() => expect(box()).toHaveFocus());
});

test("start over is shown only with turns and sits outside the composer form", async () => {
  const { user } = setup();
  expect(screen.queryByRole("button", { name: "Start over" })).not.toBeInTheDocument();
  await ask(user, "Why?");
  const startOver = await screen.findByRole("button", { name: "Start over" });
  expect(startOver.closest("form")).toBeNull();
});

test("chat request body matches the contract", async () => {
  const { user, fetchImpl } = setup();
  await ask(user, "  Why?  ");
  await screen.findByText(replyText);
  const sent = bodies(fetchImpl, "/api/chat");
  expect(sent).toHaveLength(1);
  expect(Object.keys(sent[0]).sort()).toEqual(
    ["direction", "focus_id", "history", "matches", "message", "query", "system"],
  );
  expect(sent[0]).toEqual({
    direction: "resume_to_posting",
    query: contract.match.query,
    focus_id: first.id,
    matches: contract.match.matches.map((m) => ({ id: m.id, rank: m.rank, score: m.score })),
    system: contract.match.system,
    history: [],
    message: "Why?",
  });
});

test("history carries completed turns", async () => {
  const { user, fetchImpl } = setup();
  await ask(user, "Why?");
  await screen.findByText(replyText);
  await ask(user, "More?");
  await waitFor(() => expect(bodies(fetchImpl, "/api/chat")).toHaveLength(2));
  const { history, message } = bodies(fetchImpl, "/api/chat")[1];
  expect(message).toBe("More?");
  expect(history.map((h) => h.role)).toEqual(["user", "assistant"]);
  expect(history[0].content).toBe("Why?");
});

test("failed request can be retried once", async () => {
  let fail = true;
  const fetchImpl = routedFetch({ chat: () => (fail ? reply(agent.chat_error, 502) : reply(agent.chat)) });
  const { user } = setup({ fetchImpl });
  await ask(user, "Why?");
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("The assistant could not answer");
  expect(alert).toHaveTextContent(agent.chat_error.error.message);
  expect(screen.getAllByText("Why?")).toHaveLength(1);
  fail = false;
  await user.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByText(replyText)).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  expect(screen.getAllByText("Why?")).toHaveLength(1);
  const sent = bodies(fetchImpl, "/api/chat");
  expect(sent.map((b) => b.message)).toEqual(["Why?", "Why?"]);
  expect(sent[1].history).toEqual([]);
});

test("network failure shows the unreachable title", async () => {
  const fetchImpl = vi.fn().mockRejectedValue(new TypeError("down"));
  const { user } = setup({ fetchImpl });
  await ask(user, "Why?");
  expect(await screen.findByRole("alert")).toHaveTextContent("Cannot reach the GetWork server");
  expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
});

test("chat disabled shows the reason", () => {
  seed();
  const off = { ...health, chat: { enabled: false, reason: "Set VOYAGER_API_KEY to enable chat." } };
  const fetchImpl = routedFetch();
  render(<ChatPage matchId={first.id} health={off} fetchImpl={fetchImpl} />);
  expect(screen.getByText("Set VOYAGER_API_KEY to enable chat.")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Back to results" })).toHaveAttribute("href", "#/jobs");
  expect(screen.queryByLabelText("Your question")).not.toBeInTheDocument();
  expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
  expect(screen.getByRole("heading", { level: 1 })).toHaveFocus();
  expect(fetchImpl).not.toHaveBeenCalled();
});

test("chat disabled without a reason", () => {
  for (const noChat of [{ ...health, chat: { enabled: false, reason: null } }, { ...health, chat: undefined }]) {
    const { unmount } = render(<ChatPage matchId={first.id} health={noChat} fetchImpl={routedFetch()} />);
    expect(screen.getByText("Chat is not available on this server.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Go to Find jobs" })).toHaveAttribute("href", "#/jobs");
    unmount();
  }
});

test("no session shows the open-from-a-result state", () => {
  const fetchImpl = routedFetch();
  render(<ChatPage matchId={null} health={health} fetchImpl={fetchImpl} />);
  expect(screen.getByText(NO_SESSION)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Find jobs" })).toHaveAttribute("href", "#/jobs");
  expect(screen.getByRole("link", { name: "Find candidates" })).toHaveAttribute("href", "#/candidates");
  expect(screen.queryByLabelText("Your question")).not.toBeInTheDocument();
  expect(screen.getByRole("heading", { level: 1 })).toHaveFocus();
  expect(fetchImpl).not.toHaveBeenCalled();
});

test("unknown id and corrupt snapshot", () => {
  seed();
  const view = render(<ChatPage matchId="nope" health={health} fetchImpl={routedFetch()} />);
  expect(screen.getByText(NO_SESSION)).toBeInTheDocument();
  view.unmount();
  window.sessionStorage.setItem("getwork.results.v1", "{corrupt");
  render(<ChatPage matchId={first.id} health={health} fetchImpl={routedFetch()} />);
  expect(screen.getByText(NO_SESSION)).toBeInTheDocument();
  expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
});

test("conversation survives a remount", async () => {
  const { user, fetchImpl, unmount } = setup();
  await ask(user, "Why?");
  await screen.findByText(replyText);
  expect(loadConversation(first.id)).toHaveLength(2);
  unmount();
  render(<ChatPage matchId={first.id} health={health} fetchImpl={fetchImpl} />);
  expect(screen.getByRole("log")).toHaveTextContent("You: Why?");
  expect(screen.getByText(replyText)).toBeInTheDocument();
  await ask(user, "More?");
  await waitFor(() => expect(bodies(fetchImpl, "/api/chat")).toHaveLength(2));
  expect(bodies(fetchImpl, "/api/chat")[1].history).toHaveLength(2);
});

test("start over clears the stored conversation", async () => {
  const { user } = setup();
  await ask(user, "Why?");
  await screen.findByText(replyText);
  await user.click(screen.getByRole("button", { name: "Start over" }));
  expect(screen.getByRole("log")).toBeEmptyDOMElement();
  expect(window.sessionStorage.getItem(`${CONVERSATION_PREFIX}${first.id}`)).toBeNull();
  expect(box()).toHaveFocus();
  expect(screen.queryByRole("button", { name: "Start over" })).not.toBeInTheDocument();
});

test("conversations are per match", async () => {
  const { user, unmount } = setup();
  await ask(user, "Why?");
  await screen.findByText(replyText);
  unmount();
  render(<ChatPage matchId={second.id} health={health} fetchImpl={routedFetch()} />);
  expect(screen.getByRole("log")).toBeEmptyDOMElement();
  expect(loadConversation(second.id)).toEqual([]);
  expect(loadConversation(first.id)).toHaveLength(2);
});

test("chat works when storage writes fail", async () => {
  const { user } = setup();
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("full");
  });
  await ask(user, "Why?");
  expect(await screen.findByText(replyText)).toBeInTheDocument();
});

test("back to results from the disabled state marks returning", async () => {
  seed();
  const user = userEvent.setup();
  render(<ChatPage matchId={first.id} health={{ ...health, chat: { enabled: false } }} fetchImpl={routedFetch()} />);
  expect(snapshot().returning).toBe(false);
  await user.click(screen.getByRole("link", { name: "Back to results" }));
  expect(snapshot().returning).toBe(true);
});

test("start over is disabled while the request is pending", async () => {
  const pendingChat = vi.fn(() => new Promise(() => {}));
  const { user } = setup({ fetchImpl: routedFetch({ chat: pendingChat }) });
  await ask(user, "Why?");
  expect(screen.getByRole("button", { name: "Start over" })).toBeDisabled();
});

test("a failed user turn is not stored and is absent from the next history", async () => {
  let fail = true;
  const fetchImpl = routedFetch({ chat: () => (fail ? reply(agent.chat_error, 502) : reply(agent.chat)) });
  const { user } = setup({ fetchImpl });
  await ask(user, "First?");
  await screen.findByRole("alert");
  expect(loadConversation(first.id)).toEqual([]);
  fail = false;
  await ask(user, "Second?");
  await screen.findByText(replyText);
  const sent = bodies(fetchImpl, "/api/chat");
  expect(sent.map((b) => b.message)).toEqual(["First?", "Second?"]);
  expect(sent[1].history).toEqual([]);
});

test("the focus match card shows its evidence chips", () => {
  setup();
  const card = screen.getByRole("region", { name: "The suggestion" }).querySelector("article");
  expect(first.evidence).toBeTruthy();
  expect(card.querySelectorAll(".chip").length).toBeGreaterThan(0);
});

const toggle = (name) => screen.getByRole("button", { name });

test("chat column comes first and the composer sits outside the scrolling conversation", () => {
  const { container } = setup();
  const split = container.querySelector(".chat-split");
  expect([...split.children].map((el) => el.className)).toEqual(["chat-column", "chat-suggestion-panel"]);
  expect(split.children[1]).toBe(screen.getByRole("region", { name: "The suggestion" }));
  const [scroll, form] = split.children[0].children;
  expect(scroll).toHaveClass("chat-scroll");
  expect(within(scroll).getByRole("heading", { name: "Conversation" })).toBeInTheDocument();
  expect(within(scroll).getByRole("log")).toBeInTheDocument();
  expect(form.tagName).toBe("FORM");
  expect(scroll.contains(form)).toBe(false);
  const header = container.querySelector(".chat-header");
  expect(within(header).getByRole("link", { name: "Back to results" })).toBeInTheDocument();
  expect(within(header).getAllByRole("heading")).toHaveLength(1);
  expect(within(header).getByRole("button", { name: "Hide suggestion" })).toBeInTheDocument();
});

test("the toggle hides and shows the suggestion panel", async () => {
  const { user, container } = setup();
  const button = toggle("Hide suggestion");
  expect(button).toHaveAttribute("aria-expanded", "true");
  expect(button).toHaveAttribute("aria-controls", "chat-suggestion-panel");
  expect(document.getElementById("chat-suggestion-panel")).not.toBeNull();
  await user.click(button);
  expect(button).toHaveTextContent("Show suggestion");
  expect(button).toHaveAttribute("aria-expanded", "false");
  expect(document.getElementById("chat-suggestion-panel")).toHaveAttribute("hidden");
  expect(screen.queryByRole("region", { name: "The suggestion" })).not.toBeInTheDocument();
  expect(container.querySelector(".chat-split")).toHaveClass("chat-split-solo");
  expect(button).toHaveFocus();
  await user.click(button);
  expect(button).toHaveTextContent("Hide suggestion");
  expect(screen.getByRole("region", { name: "The suggestion" })).toBeInTheDocument();
  expect(container.querySelector(".chat-split")).not.toHaveClass("chat-split-solo");
  expect(button).toHaveFocus();
});

test("the toggle is absent when chat is unavailable or there is no session", () => {
  setup({ chatHealth: { ...health, chat: { enabled: false } } });
  expect(screen.queryByRole("button", { name: /suggestion$/ })).not.toBeInTheDocument();
});

test("the toggle is absent without a session", () => {
  window.sessionStorage.clear();
  render(<ChatPage matchId="missing" health={health} fetchImpl={routedFetch()} />);
  expect(screen.queryByRole("button", { name: /suggestion$/ })).not.toBeInTheDocument();
});

test("hiding the panel keeps the explanation and makes no request", async () => {
  const { user, fetchImpl } = setup();
  await user.click(screen.getByRole("button", { name: "Why this match?" }));
  expect(await screen.findByText(contract.explain.summary)).toBeInTheDocument();
  await user.click(toggle("Hide suggestion"));
  await user.click(toggle("Show suggestion"));
  expect(screen.getByText(contract.explain.summary)).toBeVisible();
  expect(screen.getByRole("button", { name: "Why this match?" })).toHaveAttribute("aria-expanded", "true");
  expect(bodies(fetchImpl, "/api/explain")).toHaveLength(1);
  expect(fetchImpl.mock.calls.filter(([p]) => p.startsWith("/api/explain"))).toHaveLength(1);
});

test("the panel choice survives a remount", async () => {
  const { user, unmount } = setup();
  await user.click(toggle("Hide suggestion"));
  expect(window.sessionStorage.getItem(PANEL_KEY)).toBe("closed");
  unmount();
  render(<ChatPage matchId={second.id} health={health} fetchImpl={routedFetch()} />);
  expect(toggle("Show suggestion")).toHaveAttribute("aria-expanded", "false");
  expect(document.getElementById("chat-suggestion-panel")).toHaveAttribute("hidden");
});

test("an unusable panel preference is ignored", async () => {
  window.sessionStorage.setItem(PANEL_KEY, "garbage");
  const { user, unmount } = setup();
  expect(toggle("Hide suggestion")).toHaveAttribute("aria-expanded", "true");
  unmount();
  window.sessionStorage.removeItem(PANEL_KEY);
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("full");
  });
  render(<ChatPage matchId={first.id} health={health} fetchImpl={routedFetch()} />);
  expect(toggle("Hide suggestion")).toHaveAttribute("aria-expanded", "true");
  await user.click(toggle("Hide suggestion"));
  expect(toggle("Show suggestion")).toHaveAttribute("aria-expanded", "false");
});

test("the chat page offers no suggested prompts", async () => {
  const { user, container } = setup();
  const names = () => screen.getAllByRole("button").map((b) => b.textContent || b.getAttribute("aria-label"));
  expect(container.querySelector(".chat-suggestions")).toBeNull();
  expect(names().sort()).toEqual(["Hide suggestion", "Send", "Why this match?"].sort());
  await ask(user, "Why?");
  await screen.findByText(replyText);
  expect(container.querySelector(".chat-suggestions")).toBeNull();
  expect(names().sort()).toEqual(["Hide suggestion", "Send", "Start over", "Why this match?"].sort());
});

test("the greeting is not stored or sent", async () => {
  const { user, fetchImpl } = setup();
  expect(screen.getByText(GREETING)).toBeInTheDocument();
  await ask(user, "Why?");
  await screen.findByText(replyText);
  const stored = JSON.parse(window.sessionStorage.getItem(`${CONVERSATION_PREFIX}${first.id}`));
  expect(stored).toHaveLength(2);
  expect(JSON.stringify(stored)).not.toContain(GREETING);
  await ask(user, "And then?");
  await waitFor(() => expect(bodies(fetchImpl, "/api/chat")).toHaveLength(2));
  for (const body of bodies(fetchImpl, "/api/chat")) {
    expect(JSON.stringify(body.history)).not.toContain(GREETING);
  }
});

test("the disclaimer is always visible", async () => {
  let release;
  const fetchImpl = routedFetch({ chat: () => new Promise((resolve) => (release = () => resolve(reply(agent.chat)))) });
  const { user } = setup({ fetchImpl });
  const shown = () => {
    const note = document.querySelector("form.chat-composer p#chat-disclaimer");
    expect(note).toHaveTextContent(AI_DISCLAIMER);
  };
  shown();
  await ask(user, "Why?");
  await screen.findByRole("status");
  shown();
  release();
  await screen.findByText(replyText);
  shown();
});

test("the disclaimer stays visible after an error", async () => {
  const { user } = setup({ fetchImpl: routedFetch({ chat: () => reply(agent.chat_error, 502) }) });
  await ask(user, "Why?");
  await screen.findByRole("alert");
  expect(document.getElementById("chat-disclaimer")).toHaveTextContent(AI_DISCLAIMER);
});

const CANDIDATE_NAME = "Priya Nair";
const candidateResponse = {
  ...contract.match,
  direction: "posting_to_resume",
  matches: contract.match.matches.map((m, i) => ({
    ...m,
    id: `r-${i + 1}`,
    display_name: i === 0 ? CANDIDATE_NAME : `Other Name${i}`,
    document: { ...m.document, id: `r-${i + 1}`, side: "resume" },
  })),
};

function setupCandidate() {
  saveResults({ direction: "posting_to_resume", response: candidateResponse, expandedId: null });
  const fetchImpl = routedFetch();
  const user = userEvent.setup();
  render(<ChatPage matchId="r-1" health={health} fetchImpl={fetchImpl} />);
  return { user, fetchImpl };
}

test("candidate suggestion card shows the demo name, no note", () => {
  setupCandidate();
  const section = screen.getByRole("region", { name: "The suggestion" });
  expect(within(section).getByRole("heading", { level: 3, name: CANDIDATE_NAME })).toBeInTheDocument();
  expect(within(section).getByText("Demo name")).toBeInTheDocument();
  expect(within(section).getByText("Contact not available in dataset")).toBeInTheDocument();
  expect(screen.queryByText(/Candidate names are placeholders/)).not.toBeInTheDocument();
});

test("chat request body carries no display name", async () => {
  const { user, fetchImpl } = setupCandidate();
  await ask(user, "Why?");
  await screen.findByText(replyText);
  const calls = fetchImpl.mock.calls.filter(([p]) => p === "/api/chat");
  expect(calls).toHaveLength(1);
  const body = JSON.stringify(JSON.parse(calls[0][1].body));
  expect(body).not.toContain(CANDIDATE_NAME);
  expect(body).not.toContain("display_name");
});

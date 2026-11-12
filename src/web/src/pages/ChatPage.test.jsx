import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import ChatPage, { STARTERS } from "./ChatPage.jsx";
import { CONVERSATION_PREFIX, loadConversation } from "../conversationStore.js";
import contract from "../fixtures/contract.json";
import agent from "../fixtures/contract_agent.json";
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
  expect(order[0]).toBe("A");
  expect(order[1]).toBe("H1");
  expect(order.at(-1)).toBe("FORM");
});

test("back link follows the searched direction", () => {
  seed("posting_to_resume");
  render(<ChatPage matchId={first.id} health={health} fetchImpl={routedFetch()} />);
  expect(screen.getByRole("link", { name: "Back to results" })).toHaveAttribute("href", "#/candidates");
  expect(screen.getByRole("button", { name: STARTERS.posting_to_resume[0] })).toBeInTheDocument();
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
  expect(await screen.findByText(agent.chat.claims[0].text)).toBeInTheDocument();
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
  await screen.findByText(agent.chat.claims[0].text);
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
  await screen.findByText(agent.chat.claims[0].text);
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
  expect(await screen.findByText(agent.chat.claims[0].text)).toBeInTheDocument();
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
  await screen.findByText(agent.chat.claims[0].text);
  expect(loadConversation(first.id)).toHaveLength(2);
  unmount();
  render(<ChatPage matchId={first.id} health={health} fetchImpl={fetchImpl} />);
  expect(screen.getByRole("log")).toHaveTextContent("You: Why?");
  expect(screen.getByText(agent.chat.claims[0].text)).toBeInTheDocument();
  await ask(user, "More?");
  await waitFor(() => expect(bodies(fetchImpl, "/api/chat")).toHaveLength(2));
  expect(bodies(fetchImpl, "/api/chat")[1].history).toHaveLength(2);
});

test("start over clears the stored conversation", async () => {
  const { user } = setup();
  await ask(user, "Why?");
  await screen.findByText(agent.chat.claims[0].text);
  await user.click(screen.getByRole("button", { name: "Start over" }));
  expect(screen.getByRole("log")).toBeEmptyDOMElement();
  expect(window.sessionStorage.getItem(`${CONVERSATION_PREFIX}${first.id}`)).toBeNull();
  expect(box()).toHaveFocus();
  expect(screen.queryByRole("button", { name: "Start over" })).not.toBeInTheDocument();
});

test("conversations are per match", async () => {
  const { user, unmount } = setup();
  await ask(user, "Why?");
  await screen.findByText(agent.chat.claims[0].text);
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
  expect(await screen.findByText(agent.chat.claims[0].text)).toBeInTheDocument();
});

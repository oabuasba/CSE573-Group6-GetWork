import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { ApiError } from "../api.js";
import agent from "../fixtures/contract_agent.json";
import ChatThread from "./ChatThread.jsx";

const assistant = { ...agent.chat, role: "assistant" };
const asked = { role: "user", content: "Why is j-1 first?" };
const STARTERS = ["Why is this ranked here?", "Which skills am I missing?"];
const EMPTY = "Ask anything about why this suggestion fits, or what is missing.";

function setup(props = {}) {
  const handlers = { onRetry: vi.fn(), onAsk: vi.fn() };
  const view = render(
    <ChatThread turns={[]} status="idle" error={null} starters={STARTERS} {...handlers} {...props} />,
  );
  return { ...handlers, ...view };
}

afterEach(() => {
  delete Element.prototype.scrollIntoView;
});

test("renders user and assistant bubbles", async () => {
  const user = userEvent.setup();
  const { container } = setup({ turns: [asked, assistant] });
  const log = screen.getByRole("log");
  expect(log).toHaveAttribute("aria-live", "polite");
  const bubbles = log.querySelectorAll(".chat-bubble");
  expect(bubbles).toHaveLength(2);
  expect(bubbles[0]).toHaveClass("chat-bubble-user");
  expect(bubbles[0]).toHaveTextContent("You: Why is j-1 first?");
  expect(bubbles[1]).toHaveClass("chat-bubble-assistant");
  expect(bubbles[1]).toHaveTextContent("Assistant: ");
  expect(within(bubbles[1]).getByText(agent.chat.claims[0].text)).toBeInTheDocument();
  const buttons = within(bubbles[1]).getAllByRole("button", { name: "Show evidence" });
  expect(buttons).toHaveLength(2);
  await user.click(buttons[0]);
  expect(within(bubbles[1]).getByText("Shared skills: JavaScript, CSS.")).toBeInTheDocument();
  expect(container.querySelector(".chat-note")).toBeNull();
});

test("shows the note under the claims of an assistant turn", () => {
  setup({ turns: [asked, { ...assistant, note: "Part of the answer was left out." }] });
  const note = screen.getByText("Part of the answer was left out.");
  expect(note).toHaveClass("chat-note");
  expect(screen.getByRole("log")).toContainElement(note);
});

test("pending bubble while thinking", () => {
  setup({ turns: [asked], status: "thinking" });
  const pending = screen.getByRole("status");
  expect(pending).toHaveTextContent("The assistant is thinking…");
  expect(pending).toHaveClass("chat-bubble", "chat-bubble-assistant");
  expect(screen.getByRole("log")).not.toContainElement(pending);
  expect(screen.getByRole("log")).toHaveTextContent("Why is j-1 first?");
});

test("suggestions and starters send", async () => {
  const user = userEvent.setup();
  const first = setup();
  expect(screen.getByText(EMPTY)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: STARTERS[1] }));
  expect(first.onAsk).toHaveBeenCalledWith(STARTERS[1]);
  first.unmount();

  const older = { ...assistant, suggested_questions: ["Old question?"] };
  const second = setup({ turns: [asked, older, asked, assistant] });
  expect(screen.queryByText(EMPTY)).toBeNull();
  expect(screen.queryByRole("button", { name: STARTERS[0] })).toBeNull();
  expect(screen.queryByRole("button", { name: "Old question?" })).toBeNull();
  await user.click(screen.getByRole("button", { name: "What is j-1 missing?" }));
  expect(second.onAsk).toHaveBeenCalledWith("What is j-1 missing?");
});

test("suggestions are disabled while thinking", () => {
  setup({ status: "thinking" });
  expect(screen.getByRole("button", { name: STARTERS[0] })).toBeDisabled();
});

test("off-topic reply is plain text", () => {
  const offTopic = { ...assistant, off_topic: true, reply: "I can only talk about this match." };
  setup({ turns: [asked, offTopic] });
  expect(screen.getByText(offTopic.reply)).toHaveClass("measure");
  expect(screen.queryByRole("button", { name: "Show evidence" })).toBeNull();
});

test("error alert with retry", async () => {
  const user = userEvent.setup();
  const error = new ApiError({ status: 502, code: "assistant_failed", ...agent.chat_error.error });
  const { onRetry, container } = setup({ turns: [asked], status: "error", error });
  const alert = screen.getByRole("alert");
  expect(alert).toHaveTextContent("The assistant could not answer");
  expect(alert).toHaveTextContent(agent.chat_error.error.message);
  expect(alert).toHaveTextContent(agent.chat_error.error.hint);
  expect(screen.getByRole("log")).not.toContainElement(alert);
  expect(container.querySelectorAll(".chat-bubble-assistant")).toHaveLength(0);
  await user.click(screen.getByRole("button", { name: "Try again" }));
  expect(onRetry).toHaveBeenCalledOnce();
});

test("scrolls the newest item into view", () => {
  const scrollIntoView = vi.fn();
  Element.prototype.scrollIntoView = scrollIntoView;
  const base = { error: null, onRetry: vi.fn(), onAsk: vi.fn(), starters: [] };
  const { rerender } = render(<ChatThread {...base} turns={[]} status="idle" />);
  scrollIntoView.mockClear();
  rerender(<ChatThread {...base} turns={[asked]} status="idle" />);
  expect(scrollIntoView).toHaveBeenCalledTimes(1);
  rerender(<ChatThread {...base} turns={[asked]} status="thinking" />);
  expect(scrollIntoView).toHaveBeenCalledTimes(2);
  rerender(<ChatThread {...base} turns={[asked, assistant]} status="idle" />);
  expect(scrollIntoView).toHaveBeenCalledTimes(3);
  expect(scrollIntoView).toHaveBeenLastCalledWith({ block: "end" });
});

test("works without scrollIntoView", () => {
  expect(Element.prototype.scrollIntoView).toBeUndefined();
  const { rerender } = setup();
  expect(() => rerender(<ChatThread turns={[asked]} status="thinking" error={null} />)).not.toThrow();
});

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { ApiError } from "../api.js";
import agent from "../fixtures/contract_agent.json";
import ChatThread, { GREETING } from "./ChatThread.jsx";

const assistant = { role: "assistant", reply: agent.chat.reply };
const asked = { role: "user", content: "Why is j-1 first?" };

function setup(props = {}) {
  const handlers = { onRetry: vi.fn() };
  const view = render(
    <ChatThread turns={[]} status="idle" error={null} {...handlers} {...props} />,
  );
  return { ...handlers, ...view };
}

afterEach(() => {
  delete Element.prototype.scrollIntoView;
});

test("renders user and assistant bubbles as plain text", () => {
  const { container } = setup({ turns: [asked, assistant] });
  const log = screen.getByRole("log");
  expect(log).toHaveAttribute("aria-live", "polite");
  const bubbles = log.querySelectorAll(".chat-bubble");
  expect(bubbles).toHaveLength(2);
  expect(bubbles[0]).toHaveClass("chat-bubble-user");
  expect(bubbles[0]).toHaveTextContent("You: Why is j-1 first?");
  expect(bubbles[1]).toHaveClass("chat-bubble-assistant");
  expect(bubbles[1]).toHaveTextContent("Assistant: ");
  const body = bubbles[1].querySelector("p.measure.chat-reply");
  expect(body.textContent).toBe(agent.chat.reply);
  expect(bubbles[1].querySelector("button, ul, ol, a")).toBeNull();
  expect(container.querySelector(".chat-note")).toBeNull();
});

test("assistant text with markup is escaped", () => {
  const reply = "<b>x</b> https://example.com";
  setup({ turns: [asked, { role: "assistant", reply }] });
  const bubble = screen.getByRole("log").querySelector(".chat-bubble-assistant");
  expect(bubble.querySelector(".chat-reply").textContent).toBe(reply);
  expect(bubble.querySelector("b, a")).toBeNull();
});

test("has no suggestion buttons", () => {
  const { container } = setup({ turns: [asked, assistant] });
  expect(screen.queryByRole("button")).toBeNull();
  expect(container.querySelector(".chat-suggestions")).toBeNull();
  expect(setup().container.querySelector(".chat-suggestions")).toBeNull();
});

test("greeting is shown outside the log", () => {
  for (const turns of [[], [asked, assistant]]) {
    const { container, unmount } = setup({ turns });
    const greeting = container.querySelector(".chat-greeting");
    expect(greeting).toHaveClass("chat-bubble", "chat-bubble-assistant");
    expect(greeting).toHaveTextContent(`Assistant: ${GREETING}`);
    expect(screen.getByRole("log")).not.toContainElement(greeting);
    expect(container.querySelector(".chat-thread").firstElementChild).toBe(greeting);
    unmount();
  }
  expect(setup().container.querySelectorAll(".chat-turns .chat-bubble")).toHaveLength(0);
});

test("pending bubble while thinking", () => {
  setup({ turns: [asked], status: "thinking" });
  const pending = screen.getByRole("status");
  expect(pending).toHaveTextContent("The assistant is thinking…");
  expect(pending).toHaveClass("chat-bubble", "chat-bubble-assistant");
  expect(screen.getByRole("log")).not.toContainElement(pending);
  expect(screen.getByRole("log")).toHaveTextContent("Why is j-1 first?");
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
  expect(container.querySelectorAll(".chat-turns .chat-bubble-assistant")).toHaveLength(0);
  await user.click(screen.getByRole("button", { name: "Try again" }));
  expect(onRetry).toHaveBeenCalledOnce();
});

test("scrolls the newest item into view", () => {
  const scrollIntoView = vi.fn();
  Element.prototype.scrollIntoView = scrollIntoView;
  const base = { error: null, onRetry: vi.fn() };
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

test("scrolls the chat-scroll ancestor instead of the page", () => {
  const scrollIntoView = vi.fn();
  Element.prototype.scrollIntoView = scrollIntoView;
  const setTop = vi.fn();
  const props = { status: "idle", error: null, onRetry: vi.fn() };
  const { container, rerender } = render(
    <section className="chat-scroll">
      <ChatThread turns={[]} {...props} />
    </section>,
  );
  const scroller = container.querySelector(".chat-scroll");
  Object.defineProperty(scroller, "scrollHeight", { configurable: true, value: 500 });
  Object.defineProperty(scroller, "scrollTop", { configurable: true, get: () => 0, set: setTop });
  scrollIntoView.mockClear();
  rerender(
    <section className="chat-scroll">
      <ChatThread turns={[asked]} {...props} />
    </section>,
  );
  expect(setTop).toHaveBeenCalledWith(500);
  expect(scrollIntoView).not.toHaveBeenCalled();
});

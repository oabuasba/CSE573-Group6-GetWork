import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";
import ChatComposer, { AI_DISCLAIMER } from "./ChatComposer.jsx";

const box = () => screen.getByRole("textbox", { name: "Your question" });
const sendButton = () => screen.getByRole("button", { name: "Send" });

test("enter sends and shift+enter adds a line", async () => {
  const user = userEvent.setup();
  const onSend = vi.fn();
  render(<ChatComposer onSend={onSend} />);
  await user.type(box(), "a{Shift>}{Enter}{/Shift}b");
  expect(onSend).not.toHaveBeenCalled();
  expect(box()).toHaveValue("a\nb");
  await user.type(box(), "{Enter}");
  expect(onSend).toHaveBeenCalledExactlyOnceWith("a\nb");
  expect(box()).toHaveValue("");
  await user.type(box(), "  hi  ");
  await user.click(sendButton());
  expect(onSend).toHaveBeenLastCalledWith("hi");
});

test("blank and composing input is not sent", async () => {
  const user = userEvent.setup();
  const onSend = vi.fn();
  render(<ChatComposer onSend={onSend} />);
  expect(sendButton()).toBeDisabled();
  await user.type(box(), "   {Enter}");
  expect(onSend).not.toHaveBeenCalled();
  expect(sendButton()).toBeDisabled();
  await user.clear(box());
  await user.type(box(), "text");
  fireEvent.keyDown(box(), { key: "Enter", isComposing: true });
  expect(onSend).not.toHaveBeenCalled();
});

test("enter during an IME composition (keyCode 229) is not sent", async () => {
  const user = userEvent.setup();
  const onSend = vi.fn();
  render(<ChatComposer onSend={onSend} />);
  await user.type(box(), "text");
  fireEvent.keyDown(box(), { key: "Enter", keyCode: 229 });
  expect(onSend).not.toHaveBeenCalled();
});

test("rows follow the draft and the cap", () => {
  render(<ChatComposer onSend={vi.fn()} />);
  expect(box()).toHaveAttribute("rows", "1");
  fireEvent.change(box(), { target: { value: "a\n\nb" } });
  expect(box()).toHaveAttribute("rows", "3");
  fireEvent.change(box(), { target: { value: Array(10).fill("x").join("\n") } });
  expect(box()).toHaveAttribute("rows", "6");
});

test("disabled keeps the textarea usable", async () => {
  const user = userEvent.setup();
  const onSend = vi.fn();
  render(<ChatComposer onSend={onSend} disabled />);
  await user.type(box(), "hello{Enter}");
  expect(box()).toBeEnabled();
  expect(box()).toHaveValue("hello");
  expect(sendButton()).toBeDisabled();
  expect(onSend).not.toHaveBeenCalled();
});

test("roles, names and classes", () => {
  const { container } = render(<ChatComposer onSend={vi.fn()} />);
  expect(box()).toHaveAttribute("maxlength", "1000");
  expect(box()).toHaveAttribute("placeholder", "Ask about this suggestion");
  expect(box()).toHaveAttribute("aria-describedby", "chat-counter");
  expect(box().closest("div.chat-composer-box")?.closest("form.chat-composer")).not.toBeNull();
  expect(screen.getByText("0 / 1000")).toHaveClass("chat-counter");
  expect(screen.queryByRole("button", { name: "Start over" })).toBeNull();
  expect(container.querySelector("label.sr-only[for='chat-input']")).toHaveTextContent("Your question");
});

test("shows the AI disclaimer", () => {
  const { container } = render(<ChatComposer onSend={vi.fn()} />);
  const note = container.querySelector("form.chat-composer p#chat-disclaimer.chat-disclaimer");
  expect(note).toHaveTextContent(AI_DISCLAIMER);
  expect(note.textContent).toBe(AI_DISCLAIMER);
  const boxEl = container.querySelector("div.chat-composer-box");
  expect(boxEl.nextElementSibling).toBe(note);
  expect(box()).toHaveAttribute("aria-describedby", "chat-counter");
});

test("enter sends again after a composition ends", async () => {
  const user = userEvent.setup();
  const onSend = vi.fn();
  render(<ChatComposer onSend={onSend} />);
  await user.type(box(), "text");
  fireEvent.compositionStart(box());
  fireEvent.keyDown(box(), { key: "Enter", isComposing: true, keyCode: 229 });
  expect(onSend).not.toHaveBeenCalled();
  fireEvent.compositionEnd(box());
  await user.type(box(), "{Enter}");
  expect(onSend).toHaveBeenCalledExactlyOnceWith("text");
});

test("send is a round icon button inside the composer box", () => {
  const { container } = render(<ChatComposer onSend={vi.fn()} />);
  const button = sendButton();
  expect(button.closest(".chat-composer-bar")?.parentElement).toHaveClass("chat-composer-box");
  expect(box().compareDocumentPosition(button) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(button).toHaveAttribute("type", "submit");
  expect(button).toHaveClass("btn", "btn-primary", "chat-send");
  expect(button).not.toHaveClass("btn-cta");
  expect(button).toHaveAttribute("aria-label", "Send");
  expect(button.textContent).toBe("");
  const icon = button.querySelector("svg");
  expect(icon).toHaveAttribute("aria-hidden", "true");
  expect(icon).toHaveAttribute("focusable", "false");
  expect(container.querySelectorAll("button")).toHaveLength(1);
});

describe("auto-grow", () => {
  afterEach(() => {
    delete HTMLTextAreaElement.prototype.scrollHeight;
  });

  test("textarea height follows scrollHeight and resets after send", async () => {
    Object.defineProperty(HTMLTextAreaElement.prototype, "scrollHeight", {
      configurable: true,
      get() {
        return this.value.length > 0 ? 72 : 24;
      },
    });
    const user = userEvent.setup();
    render(<ChatComposer onSend={vi.fn()} />);
    expect(box().style.height).toBe("24px");
    await user.type(box(), "hello");
    expect(box().style.height).toBe("72px");
    await user.type(box(), "{Enter}");
    expect(box().style.height).toBe("24px");
  });

  test("no pixel height is set when scrollHeight is 0", async () => {
    const user = userEvent.setup();
    render(<ChatComposer onSend={vi.fn()} />);
    await user.type(box(), "hello");
    expect(box().style.height).toBe("auto");
  });
});

test("counter is quiet until near the limit", () => {
  render(<ChatComposer onSend={vi.fn()} />);
  fireEvent.change(box(), { target: { value: "x".repeat(899) } });
  const counter = screen.getByText("899 / 1000");
  expect(counter).toHaveClass("chat-counter");
  expect(counter).not.toHaveClass("chat-counter-near");
  fireEvent.change(box(), { target: { value: "x".repeat(900) } });
  expect(screen.getByText("900 / 1000")).toHaveClass("chat-counter-near");
  expect(counter).not.toHaveAttribute("aria-live");
  expect(counter).not.toHaveAttribute("role");
});

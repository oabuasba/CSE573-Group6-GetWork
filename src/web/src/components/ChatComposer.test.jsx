import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import ChatComposer from "./ChatComposer.jsx";

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
  expect(screen.getByText("0 / 1000")).toHaveClass("chat-counter");
  expect(sendButton()).toHaveClass("btn", "btn-primary");
  expect(sendButton()).not.toHaveClass("btn-cta");
  expect(sendButton()).toHaveAttribute("type", "submit");
  expect(container.querySelector("form.chat-composer")).not.toBeNull();
  expect(screen.queryByRole("button", { name: "Start over" })).toBeNull();
});

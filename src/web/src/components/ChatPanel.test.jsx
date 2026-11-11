import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import ChatPanel from "./ChatPanel.jsx";
import agent from "../fixtures/contract_agent.json";

const reply = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });

function setup(fetchImpl) {
  const user = userEvent.setup();
  const view = render(
    <ChatPanel
      direction="resume_to_posting"
      query={{ kind: "corpus", id: "r-1" }}
      focusId="j-1"
      matches={[{ id: "j-1", rank: 1, score: 0.9 }]}
      system="Full"
      fetchImpl={fetchImpl}
    />,
  );
  return { user, ...view };
}

const input = () => screen.getByLabelText("Your question");
const sendButton = () => screen.getByRole("button", { name: "Send" });

test("sends a chat request and shows the reply", async () => {
  let resolve;
  const fetchImpl = vi.fn(() => new Promise((r) => { resolve = r; }));
  const { user } = setup(fetchImpl);
  await user.type(input(), "Why?");
  await user.click(sendButton());
  expect(screen.getByRole("log")).toHaveTextContent("Why?");
  expect(screen.getByRole("status")).toHaveTextContent("Thinking...");
  expect(input()).toBeDisabled();
  expect(sendButton()).toBeDisabled();
  resolve(reply(agent.chat));
  expect(await screen.findByText(agent.chat.claims[0].text)).toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: "Show evidence" })).toHaveLength(agent.chat.claims.length);
  await waitFor(() => expect(input()).toHaveFocus());
  expect(JSON.parse(fetchImpl.mock.calls[0][1].body).message).toBe("Why?");
});

test("suggested questions send", async () => {
  const fetchImpl = vi.fn().mockResolvedValue(reply(agent.chat));
  const { user } = setup(fetchImpl);
  await user.type(input(), "Why?");
  await user.click(sendButton());
  const [question] = agent.chat.suggested_questions;
  await user.click(await screen.findByRole("button", { name: question }));
  await waitFor(() => expect(fetchImpl).toHaveBeenCalledTimes(2));
  const body = JSON.parse(fetchImpl.mock.calls[1][1].body);
  expect(body.message).toBe(question);
  expect(body.history).toHaveLength(2);
});

test("off-topic reply is plain text", async () => {
  const offTopic = { ...agent.chat, off_topic: true, claims: [], reply: "I can only discuss this match." };
  const { user, container } = setup(vi.fn().mockResolvedValue(reply(offTopic)));
  await user.type(input(), "Weather?");
  await user.click(sendButton());
  const text = await screen.findByText("I can only discuss this match.");
  expect(text).toHaveClass("measure");
  expect(container.querySelector("ol")).toBeNull();
});

test("assistant error shows message and retry", async () => {
  const { message, hint } = agent.chat_error.error;
  const fetchImpl = vi
    .fn()
    .mockResolvedValueOnce(reply(agent.chat_error, 502))
    .mockResolvedValue(reply(agent.chat));
  const { user } = setup(fetchImpl);
  await user.type(input(), "Why?");
  await user.click(sendButton());
  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("The assistant could not answer");
  expect(alert).toHaveTextContent(message);
  expect(alert).toHaveTextContent(hint);
  expect(screen.getByRole("log")).toHaveTextContent("Why?");
  expect(screen.queryByText(agent.chat.claims[0].text)).not.toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByText(agent.chat.claims[0].text)).toBeInTheDocument();
  expect(fetchImpl).toHaveBeenCalledTimes(2);
  expect(JSON.parse(fetchImpl.mock.calls[1][1].body).message).toBe("Why?");
  expect(screen.getAllByText("Why?")).toHaveLength(1);
});

test("input rules and roles", async () => {
  const { user, container } = setup(vi.fn());
  expect(sendButton()).toBeDisabled();
  expect(input()).toHaveAttribute("maxlength", "1000");
  expect(screen.getByText("0 / 1000")).toBeInTheDocument();
  await user.type(input(), "   ");
  expect(sendButton()).toBeDisabled();
  await user.type(input(), "hi");
  expect(sendButton()).toBeEnabled();
  expect(screen.getByText("5 / 1000")).toBeInTheDocument();
  expect(screen.getByRole("log")).toBeInTheDocument();
  expect(sendButton()).not.toHaveClass("btn-cta");
  expect(container.querySelectorAll(".btn-cta")).toHaveLength(0);
});

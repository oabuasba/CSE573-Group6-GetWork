import logging

import pytest
from agent_helpers import chat_request, make_agent, outside_vocab, tool_call

from getwork.agent import prompt
from getwork.agent.failure import AgentFailure
from getwork.webapp.engine import EngineError

pytestmark = pytest.mark.usefixtures("offline")

PLAIN = "Both documents list JavaScript and CSS."
OUTSIDE = "You could add Terraform and run a Cluster of workers."
OUTSIDE_LOG = "agent reply names skills outside these documents: terraform, Cluster"


def chat_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "getwork.agent"]


def test_chat_agent_answer() -> None:
    chat, fake = make_agent([PLAIN])
    response = chat.chat(chat_request())
    assert response.reply == PLAIN and response.trace == [] and response.usage.llm_calls == 1
    assert response.explainer.kind == "agent" and len(fake.calls) == 1
    assert set(response.model_dump()) == {"reply", "trace", "usage", "explainer"}


def test_chat_returns_model_text_unverified() -> None:
    chat, fake = make_agent([OUTSIDE], vocab=outside_vocab())
    response = chat.chat(chat_request())
    assert response.reply == OUTSIDE and len(fake.calls) == 1


def test_ungrounded_skills_are_logged_not_blocking(caplog: pytest.LogCaptureFixture) -> None:
    question = "ZEBRA-QUESTION-7 why is j-1 first?"
    chat, _ = make_agent([OUTSIDE], vocab=outside_vocab())
    with caplog.at_level(logging.INFO, logger="getwork.agent"):
        response = chat.chat(chat_request(message=question))
    assert response.reply == OUTSIDE
    lines = [m for m in chat_records(caplog) if "outside these documents" in m]
    assert lines == [OUTSIDE_LOG]
    assert all(question not in m and OUTSIDE not in m for m in chat_records(caplog))
    caplog.clear()
    clean, _ = make_agent([PLAIN], vocab=outside_vocab())
    with caplog.at_level(logging.INFO, logger="getwork.agent"):
        clean.chat(chat_request())
    assert not [m for m in chat_records(caplog) if "outside these documents" in m]


def test_prompt_carries_match_data() -> None:
    chat, fake = make_agent([PLAIN])
    chat.chat(chat_request())
    user = fake.calls[0].user
    for header in ("DOCUMENTS IN SCOPE", "RESULT LIST", "FACTS", "CONVERSATION", "QUESTION"):
        assert header in user
    assert "Shared skills: JavaScript, CSS." in user
    assert "Related through Front-end: the resume lists Vue; the posting asks for React." in user
    assert fake.calls[0].system == prompt.load_system_prompt("agent_v3")


def test_chat_compare_with_tools() -> None:
    compare = tool_call("compare_matches", id_a="j-1", id_b="j-2")
    chat, fake = make_agent([compare, "j-1 ranks above j-2 because of shared skills."])
    response = chat.chat(chat_request())
    assert [t.tool for t in response.trace] == ["compare_matches"] and response.trace[0].ok
    assert response.usage.tool_calls == 1 and response.usage.llm_calls == 2
    assert "f7 [comparison]" in fake.calls[1].user


def test_chat_request_errors() -> None:
    chat, _ = make_agent([])
    with pytest.raises(EngineError) as unknown_focus:
        chat.chat(chat_request(focus_id="j-404"))
    assert unknown_focus.value.code == "not_found"
    unknown_ref = [{"id": "j-1", "rank": 1, "score": 0.5}, {"id": "j-404", "rank": 2, "score": 0.1}]
    with pytest.raises(EngineError) as unknown_result:
        chat.chat(chat_request(matches=unknown_ref))
    assert unknown_result.value.code == "not_found"
    with pytest.raises(EngineError) as same_side:
        chat.chat(chat_request(focus_id="r-2"))
    assert same_side.value.code == "side_mismatch"


def test_chat_failure_raises_agent_failure() -> None:
    chat, _ = make_agent([""] * 3)
    with pytest.raises(AgentFailure) as failed:
        chat.chat(chat_request())
    assert failed.value.reason == "invalid_reply"


def test_chat_passes_trimmed_history() -> None:
    chat, fake = make_agent([PLAIN])
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(14)]
    chat.chat(chat_request(history=history))
    prompt_text = fake.calls[0].user
    assert "user: turn 4" in prompt_text and "assistant: turn 13" in prompt_text
    assert "turn 3" not in prompt_text

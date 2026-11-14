import json
from pathlib import Path

import pytest

from getwork.agent.failure import FAILURE_COPY
from getwork.agent.models import ChatResponse
from getwork.webapp import models

pytestmark = pytest.mark.usefixtures("offline")

AGENT_CONTRACT = Path(__file__).resolve().parents[1] / "web" / "src" / "fixtures" / "contract_agent.json"


def load_contract() -> dict:
    return json.loads(AGENT_CONTRACT.read_text(encoding="utf-8"))


def test_chat_and_error_sections_validate() -> None:
    data = load_contract()
    assert set(data) == {"chat", "chat_error"}
    assert set(data["chat"]) == {"reply", "trace", "usage", "explainer"}
    assert set(data["chat"]["usage"]) == {"llm_calls", "tool_calls", "elapsed_ms"}
    for item in data["chat"]["trace"]:
        assert set(item) == {"tool", "args", "ok", "error", "fact_ids"}
    chat = ChatResponse.model_validate(data["chat"])
    assert chat.reply
    assert len(chat.trace) == 1
    assert chat.explainer.kind == "agent"
    assert chat.explainer.prompt_version == "agent_v3"
    models.ErrorEnvelope.model_validate(data["chat_error"])


def test_chat_error_matches_failure_copy() -> None:
    envelope = models.ErrorEnvelope.model_validate(load_contract()["chat_error"])
    message, hint = FAILURE_COPY["llm_unavailable"]
    assert envelope.error.code == "assistant_failed"
    assert (envelope.error.message, envelope.error.hint) == (message, hint)
    assert envelope.error.field is None

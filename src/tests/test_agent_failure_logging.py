import logging

import pytest
from agent_helpers import SECRET, chat_json, make_client
from fastapi.testclient import TestClient

from getwork.agent.failure import MAX_DETAIL_CHARS, redact_detail
from getwork.llm.client import FatalLLMError, PermanentLLMError, TransientLLMError
from getwork.webapp import cli

pytestmark = pytest.mark.usefixtures("offline")

PROMPT_TEXT = "private question about my salary"
LEAKY_URL = "https://relay.example.invalid/v1/chat?api_key=abc"


def failed_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("agent failed:")]


def post_chat(script: list, caplog: pytest.LogCaptureFixture) -> tuple[TestClient, object, list[str]]:
    client, _ = make_client(script)
    with caplog.at_level(logging.INFO, logger="getwork.agent"):
        response = client.post("/api/chat", json=chat_json(message=PROMPT_TEXT))
    return client, response, failed_lines(caplog)


@pytest.mark.parametrize(
    ("script", "reason", "fragment"),
    [
        ([TransientLLMError("down", status=503)] * 2, "llm_unavailable", "TransientLLMError status=503"),
        ([FatalLLMError("denied", status=401)], "model_rejected", "FatalLLMError status=401: denied"),
        ([PermanentLLMError("reasoning_effort not allowed", status=400)], "model_rejected",
         "PermanentLLMError status=400: reasoning_effort not allowed"),
        (["{not a tool"] * 3, "invalid_reply", "last reply starts: '{not a tool'"),
    ],
)  # fmt: skip
def test_each_failure_kind_logs_one_line_and_a_502_reason(
    script: list, reason: str, fragment: str, caplog: pytest.LogCaptureFixture
) -> None:
    _, response, lines = post_chat(script, caplog)
    assert response.status_code == 502 and response.json()["error"]["reason"] == reason
    assert len(lines) == 1 and lines[0].startswith(f"agent failed: reason={reason} detail=")
    assert fragment in lines[0]
    assert PROMPT_TEXT not in " ".join(r.getMessage() for r in caplog.records)


def test_secret_key_and_url_never_reach_logs_or_response(caplog: pytest.LogCaptureFixture) -> None:
    error = PermanentLLMError(f"bad {SECRET} Bearer tok123 at {LEAKY_URL}", status=400)
    _, response, lines = post_chat([error], caplog)
    everything = " ".join(lines) + response.text
    for leaked in (SECRET, "tok123", "relay.example.invalid", "api_key=abc"):
        assert leaked not in everything
    assert len(lines) == 1


def test_secret_in_model_reply_is_redacted(caplog: pytest.LogCaptureFixture) -> None:
    _, _, lines = post_chat([f"{{{SECRET} not json"] * 3, caplog)
    assert len(lines) == 1 and SECRET not in lines[0]


def test_redact_detail_truncates_and_strips_urls() -> None:
    text = redact_detail("x " * 500 + LEAKY_URL, [SECRET])
    assert len(text) <= MAX_DETAIL_CHARS
    assert "relay.example" not in redact_detail(f"see {LEAKY_URL}", [])


def test_serve_installs_a_stderr_handler_for_the_getwork_logger(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.configure_app_logging([SECRET])
    try:
        logging.getLogger("getwork.agent").warning("agent failed: reason=x detail=%s", SECRET)
        err = capsys.readouterr().err
        assert "WARNING getwork.agent: agent failed: reason=x detail=***" in err and SECRET not in err
        cli.configure_app_logging([SECRET])
        logging.getLogger("getwork.agent").warning("once")
        assert capsys.readouterr().err.count("once") == 1
    finally:
        logging.getLogger("getwork").handlers.clear()
        logging.getLogger("getwork").setLevel(logging.NOTSET)

import json

import pytest

from getwork.llm.client import LLMRequest, TransientLLMError
from getwork.llm.fake import FakeLLMClient, make_output_json

pytestmark = pytest.mark.usefixtures("offline")


def _request(user: str = "u") -> LLMRequest:
    return LLMRequest(
        system="s", user=user, model="m", temperature=0.0, max_tokens=10,
        reasoning_effort=None, response_schema=None,
    )


def test_fake_client_script_mode() -> None:
    error = TransientLLMError("boom")
    client = FakeLLMClient(script=["first", error, "third"])
    requests = [_request(user=str(i)) for i in range(4)]
    assert client.complete(requests[0]) == "first"
    with pytest.raises(TransientLLMError) as info:
        client.complete(requests[1])
    assert info.value is error
    assert client.complete(requests[2]) == "third"
    with pytest.raises(AssertionError, match="script exhausted"):
        client.complete(requests[3])
    assert client.calls == requests


def test_fake_client_handler_mode() -> None:
    client = FakeLLMClient(handler=lambda request: request.user.upper())
    assert client.complete(_request(user="abc")) == "ABC"

    def failing(request: LLMRequest) -> str:
        raise KeyError("boom")

    failing_client = FakeLLMClient(handler=failing)
    with pytest.raises(KeyError):
        failing_client.complete(_request())
    assert len(failing_client.calls) == 1
    assert client.describe_model("m") == {"id": "m", "found": True}
    info = {"id": "m", "found": False}
    assert FakeLLMClient(script=[], model_info=info).describe_model("m") == info


def test_fake_client_requires_exactly_one_mode() -> None:
    with pytest.raises(ValueError):
        FakeLLMClient()
    with pytest.raises(ValueError):
        FakeLLMClient(handler=lambda request: "", script=["x"])


def test_make_output_json_has_five_keys() -> None:
    assert list(json.loads(make_output_json())) == [
        "title", "skills", "experience", "education", "years_experience",
    ]

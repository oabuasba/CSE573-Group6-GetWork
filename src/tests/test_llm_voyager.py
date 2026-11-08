import json
from collections.abc import Callable

import httpx2
import pytest

from getwork.llm.client import (
    FatalLLMError,
    LLMRequest,
    PermanentLLMError,
    RateLimitError,
    TransientLLMError,
)
from getwork.llm.settings import Settings
from getwork.llm.voyager import VoyagerClient

pytestmark = pytest.mark.usefixtures("offline")

KEY = "sk-test-123"
SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def make_client(
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> tuple[VoyagerClient, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def recording(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    settings = Settings(api_key=KEY, base_url="https://example.test/v1", model="m")
    http_client = httpx2.Client(transport=httpx2.MockTransport(recording))
    return VoyagerClient(settings, http_client=http_client), seen


def make_request(
    temperature: float | None = 0.2,
    reasoning_effort: str | None = "none",
    response_schema: dict | None = SCHEMA,
) -> LLMRequest:
    return LLMRequest(
        system="sys",
        user="usr",
        model="m",
        temperature=temperature,
        max_tokens=100,
        reasoning_effort=reasoning_effort,
        response_schema=response_schema,
    )


def completion(content: str | None = "hello") -> dict:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


def ok(body: dict) -> Callable[[httpx2.Request], httpx2.Response]:
    return lambda request: httpx2.Response(200, json=body)


def test_complete_request_shape() -> None:
    client, seen = make_client(ok(completion("the text")))
    assert client.complete(make_request()) == "the text"
    assert len(seen) == 1
    req = seen[0]
    assert req.method == "POST"
    assert req.url.path == "/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(req.content)
    assert body["model"] == "m"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.2
    assert body["reasoning_effort"] == "none"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True


def test_optional_request_fields_omitted_when_none() -> None:
    client, seen = make_client(ok(completion()))
    client.complete(make_request(temperature=None, reasoning_effort=None, response_schema=None))
    body = json.loads(seen[0].content)
    for key in ("temperature", "reasoning_effort", "response_format"):
        assert key not in body


@pytest.mark.parametrize(
    ("status", "headers", "expected"),
    [
        (429, {"Retry-After": "7"}, RateLimitError),
        (401, {}, FatalLLMError),
        (500, {}, TransientLLMError),
        (400, {}, PermanentLLMError),
    ],
)
def test_error_status_mapping(status: int, headers: dict, expected: type) -> None:
    client, _ = make_client(
        lambda request: httpx2.Response(status, headers=headers, json={"error": {"message": "x"}})
    )
    with pytest.raises(expected) as info:
        client.complete(make_request())
    assert info.value.status == status
    if status == 429:
        assert info.value.retry_after == 7.0


def test_connection_error_is_transient() -> None:
    def boom(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    client, seen = make_client(boom)
    with pytest.raises(TransientLLMError):
        client.complete(make_request())
    assert len(seen) == 1


def test_key_never_in_exception_text() -> None:
    body = {"error": {"message": f"bad key {KEY}"}}
    client, _ = make_client(lambda request: httpx2.Response(401, json=body))
    with pytest.raises(FatalLLMError) as info:
        client.complete(make_request())
    assert KEY not in str(info.value)


def test_describe_model_found_and_missing() -> None:
    body = {
        "object": "list",
        "data": [
            {
                "id": "m",
                "object": "model",
                "context_window": 32768,
                "supports_response_schema": True,
            }
        ],
    }
    client, seen = make_client(ok(body))
    found = client.describe_model("m")
    assert found == {
        "id": "m",
        "found": True,
        "context_window": 32768,
        "supports_response_schema": True,
    }
    assert seen[0].url.path == "/v1/models"
    assert client.describe_model("other") == {"id": "other", "found": False}


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(200, json={"choices": []}),
        httpx2.Response(200, content=b"<html>not json</html>"),
        httpx2.Response(200, json={"unexpected": KEY}),
    ],
    ids=["empty-choices", "non-json", "missing-fields"],
)
def test_unexpected_response_shapes_become_permanent_errors(response: httpx2.Response) -> None:
    client, _ = make_client(lambda request: response)
    with pytest.raises(PermanentLLMError) as info:
        client.complete(make_request())
    assert KEY not in str(info.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(200, content=b"<html>not json</html>"),
    ],
    ids=["non-json"],
)
def test_describe_model_unexpected_shapes_become_permanent_errors(
    response: httpx2.Response,
) -> None:
    client, _ = make_client(lambda request: response)
    with pytest.raises(PermanentLLMError) as info:
        client.describe_model("m")
    assert KEY not in str(info.value)

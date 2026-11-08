import json
from collections.abc import Callable
from types import SimpleNamespace

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
from getwork.llm.voyager import VoyagerClient, build_llm_client

pytestmark = pytest.mark.usefixtures("offline")

KEY = "sk-test-123"
SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


def make_client(
    handler: Callable[[httpx2.Request], httpx2.Response],
    connection_errors_fatal: bool = False,
) -> tuple[VoyagerClient, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def recording(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    settings = Settings(api_key=KEY, base_url="https://example.test/v1", model="m")
    http_client = httpx2.Client(transport=httpx2.MockTransport(recording))
    client = VoyagerClient(
        settings, http_client=http_client, connection_errors_fatal=connection_errors_fatal
    )
    return client, seen


def make_request(
    temperature: float | None = 0.2,
    reasoning_effort: str | None = "none",
    response_schema: dict | None = SCHEMA,
    extra_body: dict | None = None,
) -> LLMRequest:
    return LLMRequest(
        system="sys",
        user="usr",
        model="m",
        temperature=temperature,
        max_tokens=100,
        reasoning_effort=reasoning_effort,
        response_schema=response_schema,
        extra_body=extra_body,
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


def raise_connect_error(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("connection refused", request=request)


def test_extra_body_merged_and_reasoning_effort_field_wins() -> None:
    client, seen = make_client(ok(completion()))
    client.complete(
        make_request(extra_body={"top_k": 5, "reasoning_effort": "x"}, reasoning_effort="none")
    )
    body = json.loads(seen[0].content)
    assert body["top_k"] == 5
    assert body["reasoning_effort"] == "none"


def test_no_extra_body_sent_when_nothing_set() -> None:
    client, seen = make_client(ok(completion()))
    client.complete(make_request(reasoning_effort=None, extra_body=None))
    body = json.loads(seen[0].content)
    assert set(body) == {"model", "messages", "max_tokens", "temperature", "response_format"}


def test_connection_refused_is_fatal_when_flag_set() -> None:
    client, _ = make_client(raise_connect_error, connection_errors_fatal=True)
    with pytest.raises(FatalLLMError) as info:
        client.complete(make_request())
    assert "https://example.test/v1" in str(info.value)
    assert KEY not in str(info.value)
    with pytest.raises(FatalLLMError) as info:
        client.describe_model("m")
    assert "https://example.test/v1" in str(info.value)
    assert KEY not in str(info.value)


def test_timeout_stays_transient_when_flag_set() -> None:
    def slow(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    client, _ = make_client(slow, connection_errors_fatal=True)
    with pytest.raises(TransientLLMError):
        client.complete(make_request())
    with pytest.raises(TransientLLMError):
        client.describe_model("m")


def test_build_llm_client_local_flag_and_placeholder_key() -> None:
    seen: list[httpx2.Request] = []

    def recording(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return raise_connect_error(request)

    def stand_in(backend: str) -> SimpleNamespace:
        return SimpleNamespace(
            backend=backend, base_url="http://localhost:8000/v1", model="m", api_key="local-no-key"
        )

    http = httpx2.Client(transport=httpx2.MockTransport(recording))
    local = build_llm_client(stand_in("local"))
    local._sdk = local._sdk.with_options(http_client=http)
    with pytest.raises(FatalLLMError):
        local.complete(make_request())
    assert seen[0].headers["authorization"] == "Bearer local-no-key"

    remote = build_llm_client(stand_in("voyager"))
    remote._sdk = remote._sdk.with_options(http_client=http)
    with pytest.raises(TransientLLMError):
        remote.complete(make_request())

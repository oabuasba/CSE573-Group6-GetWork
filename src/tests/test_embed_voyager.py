import json
from collections.abc import Callable

import httpx2
import numpy as np
import pytest

from getwork.embed.client import EmbeddingError
from getwork.embed.settings import EmbedSettings
from getwork.embed.voyager_embed import VoyagerEmbeddingClient
from getwork.llm.client import (
    FatalLLMError,
    PermanentLLMError,
    TransientLLMError,
)
from getwork.llm.retry import RetryConfig

pytestmark = pytest.mark.usefixtures("offline")

KEY = "sk-test-key"
SETTINGS = EmbedSettings(
    backend="voyager",
    model="m",
    api_key=KEY,
    base_url="https://voyager.test/v1",
    batch_size=2,
)
Handler = Callable[[httpx2.Request], httpx2.Response]


def vector_for(text: str) -> list[float]:
    return [float(len(text)), 1.0, 2.0]


def embeddings_body(inputs: list[str], reverse: bool = False, drop: int = 0) -> dict:
    data = [{"index": i, "embedding": vector_for(t), "object": "embedding"}
            for i, t in enumerate(inputs)]
    if reverse:
        data.reverse()
    if drop:
        data = data[:-drop]
    return {"data": data, "model": "m", "object": "list",
            "usage": {"prompt_tokens": 1, "total_tokens": 1}}


def answer(request: httpx2.Request, **kwargs: bool | int) -> httpx2.Response:
    inputs = json.loads(request.content)["input"]
    return httpx2.Response(200, json=embeddings_body(inputs, **kwargs))


def build(
    handler: Handler, retry: RetryConfig = RetryConfig()
) -> tuple[VoyagerEmbeddingClient, list[httpx2.Request], list[float]]:
    seen: list[httpx2.Request] = []
    sleeps: list[float] = []

    def recording(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    http_client = httpx2.Client(transport=httpx2.MockTransport(recording))
    client = VoyagerEmbeddingClient(
        SETTINGS, http_client=http_client, retry=retry, sleep=sleeps.append
    )
    return client, seen, sleeps


def test_probe_sets_dim_and_request_shape() -> None:
    client, seen, _ = build(answer)
    assert client.info.dim == 3
    assert client.info.backend == "voyager" and client.info.model == "m"
    assert len(seen) == 1
    request = seen[0]
    assert request.url.path.endswith("/embeddings")
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["model"] == "m"
    assert body["input"] == ["dimension probe"]
    assert body["encoding_format"] == "float"


def test_batching_order_and_normalization() -> None:
    client, seen, _ = build(lambda r: answer(r, reverse=True))
    texts = ["a", "bbb", "cc", "dddd", "e"]
    vectors = client.embed(texts)
    assert [len(json.loads(r.content)["input"]) for r in seen[1:]] == [2, 2, 1]
    assert vectors.shape == (5, 3) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    raw = np.array([vector_for(t) for t in texts], dtype=np.float32)
    expected = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    assert np.allclose(vectors, expected, atol=1e-5)
    before = len(seen)
    assert client.embed([]).shape == (0, 3)
    assert len(seen) == before


def test_short_response_is_error() -> None:
    client, _, _ = build(lambda r: answer(r, drop=1) if len(json.loads(r.content)["input"]) > 1
                         else answer(r))
    with pytest.raises(EmbeddingError, match="expected 2"):
        client.embed(["a", "b"])


def flaky(statuses: list[int], headers: dict[str, str] | None = None) -> Handler:
    remaining = list(statuses)

    def handler(request: httpx2.Request) -> httpx2.Response:
        if remaining:
            status = remaining.pop(0)
            if status != 200:
                return httpx2.Response(status, headers=headers, json={"error": {"message": "x"}})
        return answer(request)

    return handler


def test_transient_retry_and_backoff() -> None:
    client, seen, sleeps = build(flaky([503]), RetryConfig(backoff_base_s=3.0))
    assert client.info.dim == 3
    assert sleeps == [3.0]
    assert len(seen) == 2


def test_rate_limit_waits() -> None:
    retry = RetryConfig(rate_limit_wait_s=7.0)
    _, _, sleeps = build(flaky([429]), retry)
    assert sleeps == [7.0]
    _, _, sleeps = build(flaky([429], headers={"Retry-After": "30"}), retry)
    assert sleeps == [30.0]


@pytest.mark.parametrize(
    ("status", "error"), [(401, FatalLLMError), (404, FatalLLMError), (400, PermanentLLMError)]
)
def test_permanent_and_fatal_not_retried(status: int, error: type[Exception]) -> None:
    with pytest.raises(error):
        build(flaky([status]))
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(status, json={"error": {"message": "x"}})

    with pytest.raises(error):
        build(handler)
    assert len(seen) == 1


def test_retries_exhausted_and_key_redacted() -> None:
    retry = RetryConfig(max_attempts=3, backoff_base_s=1.0)
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(503, json={"error": {"message": f"bad key {KEY}"}})

    http_client = httpx2.Client(transport=httpx2.MockTransport(handler))
    sleeps: list[float] = []
    with pytest.raises(TransientLLMError) as excinfo:
        VoyagerEmbeddingClient(SETTINGS, http_client=http_client, retry=retry, sleep=sleeps.append)
    assert len(seen) == 3 and sleeps == [1.0, 2.0]
    assert KEY not in str(excinfo.value)

    def broken(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"cannot connect with {KEY}")

    http_client = httpx2.Client(transport=httpx2.MockTransport(broken))
    with pytest.raises(TransientLLMError) as excinfo:
        VoyagerEmbeddingClient(
            SETTINGS, http_client=http_client, retry=RetryConfig(max_attempts=2),
            sleep=sleeps.append,
        )
    assert KEY not in str(excinfo.value)

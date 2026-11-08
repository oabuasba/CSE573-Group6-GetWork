import pytest

from getwork.llm import client as llm_client
from getwork.llm.client import (
    FatalLLMError,
    PermanentLLMError,
    RateLimitError,
    TransientLLMError,
)

pytestmark = pytest.mark.usefixtures("offline")

SECRET = "sk-super-secret-key"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (None, TransientLLMError),
        (408, TransientLLMError),
        (500, TransientLLMError),
        (502, TransientLLMError),
        (503, TransientLLMError),
        (504, TransientLLMError),
        (429, RateLimitError),
        (401, FatalLLMError),
        (403, FatalLLMError),
        (404, FatalLLMError),
        (410, FatalLLMError),
        (400, PermanentLLMError),
        (413, PermanentLLMError),
        (422, PermanentLLMError),
        (418, PermanentLLMError),
    ],
)
def test_error_from_http_status_table(status: int | None, expected: type) -> None:
    err = llm_client.error_from_http(status=status, headers=None, message="boom")
    assert type(err) is expected
    assert err.status == status
    assert str(err) == "boom"


def test_error_from_http_retry_after() -> None:
    def retry_after(headers: dict | None) -> float | None:
        return llm_client.error_from_http(status=429, headers=headers, message="m").retry_after

    assert retry_after({"Retry-After": "7"}) == 7.0
    assert retry_after({"retry-after": "7"}) == 7.0
    assert retry_after({"Retry-After": "abc"}) is None
    assert retry_after({"Retry-After": "-3"}) is None
    assert retry_after(None) is None
    assert retry_after({}) is None


def test_error_from_http_retry_after_capped() -> None:
    err = llm_client.error_from_http(status=429, headers={"Retry-After": "999"}, message="m")
    assert err.retry_after == 300.0


def test_error_from_http_redacts_and_truncates() -> None:
    message = "x" * 1000 + SECRET + "y" * 1000
    err = llm_client.error_from_http(status=500, headers=None, message=message, secrets=[SECRET])
    assert SECRET not in str(err)
    assert len(str(err)) <= 500

    boundary = "x" * (500 - 5) + SECRET
    err = llm_client.error_from_http(status=500, headers=None, message=boundary, secrets=[SECRET])
    assert SECRET[:5] not in str(err)


def test_redact_secrets_ignores_empty_secret() -> None:
    assert llm_client.redact_secrets("hello", ["", "ell"]) == "h***o"
    assert llm_client.redact_secrets("hello", [""]) == "hello"


def test_redact_secrets_longest_first() -> None:
    assert llm_client.redact_secrets("xabcdefx", ["abc", "abcdef"]) == "x***x"


def test_describe_exception_redacts_and_truncates() -> None:
    short = llm_client.describe_exception(ValueError(f"bad {SECRET}"), secrets=[SECRET])
    assert short == "ValueError: bad ***"

    long = llm_client.describe_exception(
        RuntimeError("z" * 2000 + SECRET), secrets=[SECRET]
    )
    assert long.startswith("RuntimeError: ")
    assert len(long) <= 500
    assert SECRET not in long

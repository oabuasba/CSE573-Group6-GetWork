import json
from pathlib import Path

import pytest

from getwork.llm.client import LLMRequest, PermanentLLMError, RateLimitError, TransientLLMError
from getwork.llm.fake import FakeLLMClient
from getwork.llm.retry import RetryConfig
from getwork.relabel import reextract, unseen
from getwork.relabel.cache import RelabelProvenanceError
from getwork.relabel.llm_relabeler import LLMRelabeler
from getwork.taxonomy.model import Cluster, Taxonomy, make_taxonomy

pytestmark = pytest.mark.usefixtures("offline")

RETRY = RetryConfig(max_attempts=2, backoff_base_s=0.0, rate_limit_wait_s=0.0)
REEXTRACT_REPLY = json.dumps({"additional_skills": [{"skill": "CI/CD", "cluster": "Cloud"}]})


def _tax(version_seed: str = "AWS") -> Taxonomy:
    return make_taxonomy(
        [Cluster(0, "Cloud", (version_seed,)), Cluster(1, "Front-end", ("React",))],
        noise=(),
        alias_table_version="sha256:t",
        meta={},
    )


def _handler(request: LLMRequest) -> str:
    if request.system == reextract.load_reextract_prompt():
        return REEXTRACT_REPLY
    block = request.user.split("\n\nSKILLS:\n")[1]
    skills = [line[2:] for line in block.splitlines()]
    answers = [{"skill": s, "cluster": "Cloud"} for s in skills]
    return json.dumps({"assignments": answers})


def _make(client: FakeLLMClient, tmp_path: Path, **kw: object) -> LLMRelabeler:
    options = {"model": "m", "backend": "voyager", "cache_dir": tmp_path / "cache",
               "retry": RETRY, "sleep": lambda _s: None, **kw}
    tax = options.pop("tax", None) or _tax()
    return LLMRelabeler(client, tax, **options)


def test_relabeler_modes_and_stats(tmp_path: Path) -> None:
    def handler(request: LLMRequest) -> str:
        if "bad" in request.user and request.system == unseen.load_unseen_prompt():
            return "garbage"
        return _handler(request)

    client = FakeLLMClient(handler=handler)
    relabeler = _make(client, tmp_path, batch_size=1, extra_body={"k": 1})
    assert relabeler.mode == "llm" and LLMRelabeler.mode == "llm"
    assert (relabeler.model, relabeler.backend) == ("m", "voyager")
    assert relabeler.prompt_version == "unseen_v1+reextract_v1"
    assert relabeler.stats.calls == 0 and relabeler.failed_skills == []
    assert relabeler.assign_unseen(["bad", "good"]) == {"good": "Cloud"}
    assert relabeler.failed_skills == ["bad"]
    assert relabeler.stats.calls == 4  # 3 validation attempts for bad, 1 for good
    assert relabeler.assign_unseen(["bad2"]) == {}
    assert relabeler.stats.calls == 7
    assert relabeler.failed_skills == ["bad", "bad2"]
    assert relabeler.reextract("r-1", "text", ["React"]) == [("CI/CD", "Cloud")]
    assert relabeler.stats.calls == 8
    assert all(c.extra_body == {"k": 1} for c in client.calls)
    cache_dir = tmp_path / "cache"
    unseen_saved = json.loads((cache_dir / "unseen_assignments.json").read_text())["entries"]
    assert list(unseen_saved) == ["good"]
    assert unseen_saved["good"]["prompt_version"] == "unseen_v1"
    assert unseen_saved["good"]["taxonomy_version"] == _tax().version
    assert (unseen_saved["good"]["model"], unseen_saved["good"]["backend"]) == ("m", "voyager")
    saved = json.loads((cache_dir / "reextract.json").read_text())["entries"]
    assert len(saved) == 1
    assert next(iter(saved.values()))["prompt_version"] == "reextract_v1"


def test_relabeler_resume_from_warm_cache(tmp_path: Path) -> None:
    first = _make(FakeLLMClient(handler=_handler), tmp_path)
    assert first.assign_unseen(["Rust", "Go"]) == {"Rust": "Cloud", "Go": "Cloud"}
    assert first.reextract("r-1", "text", ["React"]) == [("CI/CD", "Cloud")]
    fresh = FakeLLMClient(script=[])
    second = _make(fresh, tmp_path)
    assert second.assign_unseen(["Go", "Rust"]) == {"Go": "Cloud", "Rust": "Cloud"}
    assert second.reextract("r-1", "text", ["React"]) == [("CI/CD", "Cloud")]
    assert fresh.calls == [] and second.stats.calls == 0


def test_relabeler_refuses_other_model(tmp_path: Path) -> None:
    first = _make(FakeLLMClient(handler=_handler), tmp_path)
    first.assign_unseen(["Rust"])
    first.reextract("r-1", "text", ["React"])
    client = FakeLLMClient(handler=_handler)
    with pytest.raises(RelabelProvenanceError) as exc:
        _make(client, tmp_path, model="other")
    assert "model" in str(exc.value)
    with pytest.raises(RelabelProvenanceError) as exc2:
        _make(client, tmp_path, backend="local")
    assert "backend" in str(exc2.value)
    mixed = _make(client, tmp_path, model="other", allow_mixed=True)
    assert mixed.model == "other"
    assert mixed.assign_unseen(["Rust"]) == {"Rust": "Cloud"} and client.calls == []
    assert mixed.reextract("r-1", "text", ["React"]) == [("CI/CD", "Cloud")]
    assert client.calls == []
    only_reextract = tmp_path / "only"
    seeded = _make(FakeLLMClient(handler=_handler), tmp_path, cache_dir=only_reextract)
    seeded.reextract("r-1", "text", ["React"])
    with pytest.raises(RelabelProvenanceError):
        _make(client, tmp_path, cache_dir=only_reextract, model="other")
    assert _make(client, tmp_path, cache_dir=only_reextract, model="other", allow_mixed=True)


def test_relabeler_other_taxonomy_drops_cache(tmp_path: Path) -> None:
    _make(FakeLLMClient(handler=_handler), tmp_path).assign_unseen(["Rust"])
    client = FakeLLMClient(handler=_handler)
    other = _make(client, tmp_path, tax=_tax("GCP"))
    assert other.assign_unseen(["Rust"]) == {"Rust": "Cloud"}
    assert len(client.calls) == 1


def test_relabeler_stats_sum_retries_and_rate_limits(tmp_path: Path) -> None:
    unseen_reply = json.dumps({"assignments": [{"skill": "Rust", "cluster": "Cloud"}]})
    client = FakeLLMClient(script=[
        RateLimitError("slow down", status=429), unseen_reply,
        TransientLLMError("blip"), REEXTRACT_REPLY,
    ])
    relabeler = _make(client, tmp_path)
    assert relabeler.assign_unseen(["Rust"]) == {"Rust": "Cloud"}
    assert (relabeler.stats.calls, relabeler.stats.retries, relabeler.stats.rate_limited) == (
        2, 1, 1)
    assert relabeler.reextract("r-1", "text", []) == [("CI/CD", "Cloud")]
    assert (relabeler.stats.calls, relabeler.stats.retries, relabeler.stats.rate_limited) == (
        4, 2, 1)


def test_relabeler_max_input_chars_cuts_text(tmp_path: Path) -> None:
    client = FakeLLMClient(handler=_handler)
    relabeler = _make(client, tmp_path, max_input_chars=5)
    relabeler.reextract("r-1", "abcdefghij", ["React"])
    assert client.calls[0].user.endswith("\n\nTEXT:\nabcde")
    relabeler.reextract("r-1", "abcdeXXXXX", ["React"])  # same cut text: cached
    assert len(client.calls) == 1


def test_relabeler_retry_and_sleep_reach_both_steps(tmp_path: Path) -> None:
    sleeps: list[float] = []
    retry = RetryConfig(max_attempts=2, backoff_base_s=0.5, rate_limit_wait_s=0.0)
    transient = TransientLLMError("blip")
    unseen_reply = json.dumps({"assignments": [{"skill": "Rust", "cluster": "Cloud"}]})
    client = FakeLLMClient(script=[transient, unseen_reply, transient, REEXTRACT_REPLY])
    relabeler = _make(client, tmp_path, retry=retry, sleep=sleeps.append)
    assert relabeler.assign_unseen(["Rust"]) == {"Rust": "Cloud"}
    assert len(sleeps) == 1
    assert relabeler.reextract("r-1", "text", []) == [("CI/CD", "Cloud")]
    assert len(sleeps) == 2 and all(s > 0 for s in sleeps)
    # max_attempts reaches both steps: with one attempt a transient error is final
    single = RetryConfig(max_attempts=1, backoff_base_s=0.5)
    both = FakeLLMClient(script=[transient, transient])
    other = _make(both, tmp_path, retry=single, cache_dir=tmp_path / "c2", sleep=sleeps.append)
    assert other.assign_unseen(["Rust"]) == {} and other.failed_skills == ["Rust"]
    assert other.reextract("r-1", "text", []) is None
    assert len(both.calls) == 2


def test_relabeler_secret_in_error_does_not_leak(tmp_path: Path) -> None:
    secret = "sk-very-secret-123"

    def handler(request: LLMRequest) -> str:
        raise PermanentLLMError(f"rejected key {secret}", status=400)

    relabeler = _make(FakeLLMClient(handler=handler), tmp_path, secrets=[secret])
    assert relabeler.assign_unseen(["Rust"]) == {}
    assert relabeler.reextract("r-1", "text", []) is None
    assert relabeler.failed_skills == ["Rust"]
    leaked = [p.name for p in (tmp_path / "cache").rglob("*") if p.is_file()
              and secret in p.read_text()]
    assert leaked == [] and secret not in repr(relabeler.failed_skills)

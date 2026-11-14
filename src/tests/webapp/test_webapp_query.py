import logging
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from artifact_helpers import build_artifacts

from getwork.embed.fake import FakeEmbeddingClient
from getwork.llm.client import FatalLLMError
from getwork.llm.settings import ConfigError
from getwork.llm.fake import FakeLLMClient, make_output_json
from getwork.structure.extract import ExtractConfig
from getwork.webapp import models
from getwork.webapp.artifacts import ArtifactEngine, EmbedResources, StructureResources
from getwork.webapp.engine import EngineError

pytestmark = pytest.mark.usefixtures("offline")

SECRET = "sk-secret-123"
SYSTEMS = ("Full", "B2", "B1+")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "art"
    build_artifacts(path)
    return path


def fake_embed(counter: list[int] | None = None) -> EmbedResources:
    def factory() -> FakeEmbeddingClient:
        if counter is not None:
            counter.append(1)
        return FakeEmbeddingClient(dim=16)

    return EmbedResources(factory=factory, describe="fake, 16")


def make_engine(root: Path, **kwargs: object) -> ArtifactEngine:
    kwargs.setdefault("embed", fake_embed())
    return ArtifactEngine.load(root, **kwargs)  # type: ignore[arg-type]


def structure_with(client: FakeLLMClient) -> StructureResources:
    return StructureResources(
        client=client, cfg=ExtractConfig(model="fake-model", backend="local"), secrets=(SECRET,)
    )


def request(direction: str, query: dict, system: str = "Full", k: int = 10) -> models.MatchRequest:
    return models.MatchRequest.model_validate(
        {"direction": direction, "query": query, "system": system, "k": k}
    )


def form(**fields: object) -> dict:
    return {"kind": "form", **fields}


def doc_form(engine: ArtifactEngine, doc_id: str) -> dict:
    """A form that reproduces the fields of a corpus document."""
    view = engine.get_document(doc_id)
    fields: dict = {"title": view.title, "skills": [s.skill for s in view.skills]}
    if view.side == "resume":
        fields["years_experience"] = view.years_experience
    else:
        fields["level"] = view.level
    return form(**fields)


@pytest.mark.parametrize("system", SYSTEMS)
@pytest.mark.parametrize(("doc_id", "direction"), [
    ("r-1", "resume_to_posting"), ("r-3", "resume_to_posting"), ("p-3", "posting_to_resume"),
])
def test_form_reproduces_corpus_ranking(
    root: Path, system: str, doc_id: str, direction: str
) -> None:
    engine = make_engine(root)
    expected = engine.match(request(direction, {"kind": "corpus", "id": doc_id}, system))
    got = engine.match(request(direction, doc_form(engine, doc_id), system))
    assert [m.id for m, _ in got.matches] == [m.id for m, _ in expected.matches]
    for (_, a), (_, b) in zip(got.matches, expected.matches, strict=True):
        assert a == pytest.approx(b, abs=1e-5)
    assert got.query.id is None and got.query.level == expected.query.level


def test_form_canonicalizes_and_labels(root: Path) -> None:
    engine = make_engine(root)
    result = engine.match(request(
        "resume_to_posting", form(title="Dev", skills=[" js ", "py", "Foo", ""], years_experience=1.0),
    ))
    skills = {s.skill: (s.cluster, s.via) for s in result.query.skills}
    assert skills == {
        "JavaScript": ("Other", "default"), "Python": ("Data", "taxonomy"),
        "Foo": ("Other", "default"),
    }
    assert result.query.id is None and result.query.side == "resume"
    resume = engine.match(request(
        "resume_to_posting", form(title="Dev", level="Director", years_experience=1.0, skills=["py"]),
    )).query
    assert (resume.level, resume.level_source) == ("Entry level", "years")
    posting = engine.match(request(
        "posting_to_resume", form(title="Dev", level="Director", skills=["py"]),
    )).query
    assert (posting.level, posting.level_source, posting.side) == ("Director", "listed", "posting")


def test_form_capability_and_mismatch_errors(root: Path) -> None:
    no_embed = ArtifactEngine.load(root, embed_reason="Disabled with --no-embed.")
    body = form(title="Dev", skills=["py"])
    with pytest.raises(EngineError) as info:
        no_embed.match(request("resume_to_posting", body, "Full"))
    assert info.value.code == "capability_unavailable"
    assert "Disabled with --no-embed." in info.value.message
    assert no_embed.match(request("resume_to_posting", body, "B1+")).matches
    with pytest.raises(EngineError) as info:
        no_embed.match(request("resume_to_posting", form(), "B1+"))
    assert info.value.code == "empty_query"
    other = EmbedResources(lambda: FakeEmbeddingClient(dim=16, model="other"), "other")
    with pytest.raises(EngineError) as info:
        make_engine(root, embed=other).match(request("resume_to_posting", body, "Full"))
    assert (info.value.code, info.value.status) == ("embedding_mismatch", 409)

    def broken() -> FakeEmbeddingClient:
        raise RuntimeError("cannot load")

    with pytest.raises(EngineError) as info:
        make_engine(root, embed=EmbedResources(broken, "broken")).match(
            request("resume_to_posting", body, "B2")
        )
    assert info.value.code == "capability_unavailable"
    assert "cannot load" not in info.value.message and "RuntimeError" in info.value.message

    def misconfigured() -> FakeEmbeddingClient:
        raise ConfigError("EMBED_MODEL is not set")

    with pytest.raises(EngineError) as info:
        make_engine(root, embed=EmbedResources(misconfigured, "x")).match(
            request("resume_to_posting", body, "B2")
        )
    assert "EMBED_MODEL is not set" in info.value.message


def test_query_notes(root: Path) -> None:
    engine = make_engine(root)

    def codes(**fields: object) -> list[str]:
        result = engine.match(request("resume_to_posting", form(**fields), "B1+"))
        return [n.code for n in result.notes]

    assert codes(title="Dev") == ["no_skills", "relabel_mode_differs"]
    assert codes(title="Dev", skills=["Python", "SQL"]) == ["sparse_query", "relabel_mode_differs"]
    assert codes(title="Dev", skills=["Foo", "Bar", "Baz"]) == [
        "sparse_query", "unknown_skills", "relabel_mode_differs",
    ]
    assert codes(title="Dev", skills=["Python", "SQL", "AWS"]) == ["relabel_mode_differs"]


def structured_client(**overrides: object) -> FakeLLMClient:
    fields = {"title": "Data Engineer", "skills": ["py", "SQL", "Foo"], "years_experience": 1.0}
    return FakeLLMClient(handler=lambda req: make_output_json(**{**fields, **overrides}))


def test_text_query_pipeline(root: Path) -> None:
    client = structured_client()
    engine = make_engine(root, structure=structure_with(client))
    text = request("resume_to_posting", {"kind": "text", "text": "I build data pipelines."})
    result = engine.match(text)
    assert len(client.calls) == 1 and client.calls[0].model == "fake-model"
    assert result.query.title == "Data Engineer" and result.query.id is None
    assert [(s.skill, s.cluster) for s in result.query.skills] == [
        ("Python", "Data"), ("SQL", "Data"), ("Foo", "Other"),
    ]
    same = engine.match(request(
        "resume_to_posting", form(title="Data Engineer", skills=["py", "SQL", "Foo"], years_experience=1.0),
    ))
    assert [(m.id, s) for m, s in result.matches] == [(m.id, s) for m, s in same.matches]


def test_text_query_failures(root: Path) -> None:
    body = {"kind": "text", "text": "Some resume text"}
    invalid = FakeLLMClient(handler=lambda req: f"not json {SECRET}")
    engine = make_engine(root, structure=structure_with(invalid))
    with pytest.raises(EngineError) as info:
        engine.match(request("resume_to_posting", body))
    assert (info.value.code, info.value.status) == ("structuring_failed", 502)
    assert SECRET not in info.value.message

    def reject(req: object) -> str:
        raise FatalLLMError(f"bad key {SECRET}", status=401)

    fatal = make_engine(root, structure=structure_with(FakeLLMClient(handler=reject)))
    with pytest.raises(EngineError) as info:
        fatal.match(request("resume_to_posting", body))
    assert (info.value.code, info.value.status) == ("structuring_failed", 502)
    assert info.value.message == "The model service rejected the request"

    off = make_engine(root, structure_reason="Disabled with --no-structure.")
    with pytest.raises(EngineError) as info:
        off.match(request("resume_to_posting", body))
    assert info.value.code == "capability_unavailable"
    assert "Disabled with --no-structure." in info.value.message


def test_text_query_limits_and_privacy(root: Path, caplog: pytest.LogCaptureFixture) -> None:
    client = structured_client()
    engine = make_engine(root, structure=structure_with(client), max_query_chars=50)
    with pytest.raises(EngineError) as info:
        engine.match(request("resume_to_posting", {"kind": "text", "text": "   "}))
    assert info.value.code == "empty_query"
    with pytest.raises(EngineError) as info:
        engine.match(request("resume_to_posting", {"kind": "text", "text": "x" * 51}))
    assert info.value.code == "text_too_long" and "50" in info.value.message
    assert client.calls == []
    secret_text = "ZEBRA-PRIVATE-RESUME-TEXT"
    with caplog.at_level(logging.DEBUG):
        engine.match(request("resume_to_posting", {"kind": "text", "text": secret_text}))
        engine.match(request("resume_to_posting", form(title="ZEBRA-FORM-TITLE", skills=["py"])))
    assert secret_text not in caplog.text and "ZEBRA-FORM-TITLE" not in caplog.text
    assert "kind=text" in caplog.text


class RecordingEmbedder(FakeEmbeddingClient):
    def __init__(self) -> None:
        super().__init__(dim=16)
        self.running = 0
        self.max_running = 0
        self._guard = threading.Lock()

    def embed(self, texts: list[str]) -> np.ndarray:
        with self._guard:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        time.sleep(0.05)
        try:
            return super().embed(texts)
        finally:
            with self._guard:
                self.running -= 1


def test_embedder_lazy_once_and_serialized(root: Path) -> None:
    built: list[RecordingEmbedder] = []

    def factory() -> RecordingEmbedder:
        built.append(RecordingEmbedder())
        return built[0] if len(built) == 1 else built[-1]

    engine = make_engine(root, embed=EmbedResources(factory=factory, describe="rec"))
    assert built == []
    body = request("resume_to_posting", form(title="Dev", skills=["py"]))
    threads = [threading.Thread(target=engine.match, args=(body,)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    engine.match(body)
    assert len(built) == 1
    assert built[0].max_running == 1 and len(built[0].calls) == 3


def test_failed_factory_is_retried_not_cached(root: Path) -> None:
    attempts: list[int] = []

    def flaky() -> FakeEmbeddingClient:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("not yet")
        return FakeEmbeddingClient(dim=16)

    engine = make_engine(root, embed=EmbedResources(flaky, "flaky"))
    body = request("resume_to_posting", form(title="Dev", skills=["py"]))
    with pytest.raises(EngineError):
        engine.match(body)
    assert engine.match(body).matches and len(attempts) == 2

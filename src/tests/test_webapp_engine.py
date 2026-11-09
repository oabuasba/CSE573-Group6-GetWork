import pytest

from getwork.docembed import levels
from getwork.webapp import demo, engine, explain, models

pytestmark = pytest.mark.usefixtures("offline")


def test_engine_error_codes() -> None:
    err = engine.EngineError("not_found", "No such id", hint="Pick another", field="query.id")
    assert (err.code, err.message, err.hint, err.field) == (
        "not_found",
        "No such id",
        "Pick another",
        "query.id",
    )
    assert err.status == models.ERROR_CODES["not_found"] == 404
    assert str(err) == "No such id"
    plain = engine.EngineError("internal", "boom")
    assert (plain.hint, plain.field, plain.status) == (None, None, 500)
    for code, status in models.ERROR_CODES.items():
        assert engine.EngineError(code, "m").status == status
    with pytest.raises(ValueError):
        engine.EngineError("nope", "m")


def request(direction: str, query: dict, system: str = "Full", k: int = 10) -> models.MatchRequest:
    return models.MatchRequest.model_validate(
        {"direction": direction, "query": query, "system": system, "k": k}
    )


def corpus(doc_id: str, system: str = "Full", k: int = 10) -> models.MatchRequest:
    return request("resume_to_posting", {"kind": "corpus", "id": doc_id}, system, k)


def test_demo_health_examples_documents() -> None:
    eng = demo.DemoEngine()
    health = eng.health()
    assert health.mode == "demo" and health.artifacts_dir is None
    assert health.counts == {"resumes": 6, "postings": 6}
    assert health.systems == ["B1+", "B2", "Full"]
    assert health.levels == [*levels.LEVELS, levels.NOT_SPECIFIED]
    assert health.versions.demo is True
    structure, embed = health.capabilities["structure"], health.capabilities["embed"]
    assert (structure.available, structure.reason) == (False, "Demo mode has no language model.")
    assert embed.available is True
    assert [v.id for v in eng.examples("resume", 6)] == [f"r-{i}" for i in range(1, 7)]
    assert [v.id for v in eng.examples("posting", 2)] == ["j-1", "j-2"]
    assert eng.get_document("j-3").side == "posting"
    with pytest.raises(engine.EngineError) as caught:
        eng.get_document("r-99")
    assert caught.value.code == "not_found" and "r-99" in caught.value.message


def test_demo_content_requirements() -> None:
    eng = demo.DemoEngine()
    resumes, postings = eng.examples("resume", 6), eng.examples("posting", 6)
    by_id = {v.id: v for v in resumes + postings}
    assert [(s.skill, s.cluster) for s in by_id["r-1"].skills] == [
        ("Vue", "Front-end"),
        ("CSS", "Front-end"),
        ("JavaScript", "Front-end"),
    ]
    assert by_id["r-1"].title == "Front-end developer"
    assert by_id["j-1"].title == "React developer"
    assert {(s.skill, s.cluster) for s in by_id["j-1"].skills} == {
        ("React", "Front-end"),
        ("JavaScript", "Front-end"),
        ("CSS", "Front-end"),
    }
    assert by_id["j-2"].title == "Java backend engineer"
    assert [(s.skill, s.cluster) for s in by_id["j-2"].skills] == [
        ("Java", "Back-end"),
        ("Spring", "Back-end"),
        ("SQL", "Data"),
    ]
    assert "Java" in by_id["r-2"].title
    assert all(3 <= len(v.skills) <= 6 for v in by_id.values())
    for docs in (resumes, postings):
        assert any(s.cluster == "Other" for v in docs for s in v.skills)
    relations = {
        explain.build_evidence("resume_to_posting", r, p).level.relation
        for r in resumes
        for p in postings
    }
    assert {"adjacent", "different"} <= relations


def test_demo_corpus_match_ranking_and_partial_credit() -> None:
    eng = demo.DemoEngine()
    full = eng.match(corpus("r-1", "Full", 4))
    assert full.k_effective == 4 and len(full.matches) == 4
    assert full.query.id == "r-1"
    scores = [score for _, score in full.matches]
    assert scores == sorted(scores, reverse=True)
    tied = [view.id for view, score in full.matches if score == 0.0]
    assert tied == sorted(tied)
    assert eng.match(corpus("r-1", k=50)).k_effective == 6
    ids = [view.id for view, _ in full.matches]
    assert ids[0] == "j-1"
    b2 = dict((v.id, s) for v, s in eng.match(corpus("r-1", "B2")).matches)
    full_scores = dict((v.id, s) for v, s in eng.match(corpus("r-1", "Full")).matches)
    assert (full_scores["j-1"], b2["j-1"]) == (0.8333, 0.6667)
    j1 = eng.get_document("j-1")
    evidence = explain.build_evidence("resume_to_posting", full.query, j1)
    assert [g.cluster for g in evidence.related] == ["Front-end"]
    assert provenance_is_demo(full)


def provenance_is_demo(result: engine.MatchResult) -> bool:
    prov = result.provenance
    return prov.demo and prov.embedding is None and prov.taxonomy_version == "demo"


def test_demo_form_match() -> None:
    eng = demo.DemoEngine()
    form = {"kind": "form", "title": "Web dev", "skills": ["js", "vue", "Figma"]}
    result = eng.match(request("resume_to_posting", form))
    assert result.query.id is None and result.query.side == "resume"
    assert [(s.skill, s.cluster, s.via) for s in result.query.skills] == [
        ("JavaScript", "Front-end", "taxonomy"),
        ("Vue", "Front-end", "taxonomy"),
        ("Figma", "Other", "default"),
    ]
    assert result.matches[0][0].id == "j-1"
    posting_form = {"kind": "form", "skills": ["k8s"], "level": "entry"}
    posting_result = eng.match(request("posting_to_resume", posting_form))
    assert posting_result.query.side == "posting" and posting_result.query.level == "Entry level"
    assert posting_result.matches[0][0].id == "r-4"
    sparse = eng.match(request("resume_to_posting", {"kind": "form", "skills": ["Figma"]}))
    assert [n.code for n in sparse.notes] == ["sparse_query", "unknown_skills"]


def test_demo_match_errors() -> None:
    eng = demo.DemoEngine()

    def code_of(req: models.MatchRequest) -> str:
        with pytest.raises(engine.EngineError) as caught:
            eng.match(req)
        return caught.value.code

    empty = request("resume_to_posting", {"kind": "form", "title": " ", "skills": [" "]})
    assert code_of(empty) == "empty_query"
    assert code_of(corpus("j-1")) == "side_mismatch"
    assert code_of(corpus("r-99")) == "not_found"
    assert code_of(request("resume_to_posting", {"kind": "text", "text": "hello"})) == (
        "capability_unavailable"
    )

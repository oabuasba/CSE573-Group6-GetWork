import pytest
from pydantic import ValidationError

from getwork.webapp import models

pytestmark = pytest.mark.usefixtures("offline")

FORM = {"kind": "form", "title": "Dev", "skills": ["Python"]}


def request(**overrides: object) -> dict:
    return {"direction": "resume_to_posting", "query": FORM, **overrides}


def doc(side: str = "resume") -> models.DocumentView:
    skill = models.DocSkill(skill="Python", cluster="Programming", via="taxonomy")
    return models.DocumentView(
        id="r-1", side=side, title="Dev", level="Associate", level_source="years",
        years_experience=3.0, skills=[skill],
    )


def test_match_request_variants() -> None:
    corpus = models.MatchRequest.model_validate(request(query={"kind": "corpus", "id": "r-1"}))
    assert isinstance(corpus.query, models.CorpusQuery)
    assert corpus.k == 10 and corpus.system == "Full"
    text = models.MatchRequest.model_validate(request(query={"kind": "text", "text": "hi"}))
    assert isinstance(text.query, models.TextQuery) and text.query.title is None
    form = models.MatchRequest.model_validate(request())
    assert isinstance(form.query, models.FormQuery)
    for k in (1, 50):
        assert models.MatchRequest.model_validate(request(k=k)).k == k
    assert models.MatchRequest.model_validate(request(system="B1+")).system == "B1+"
    edge = {**FORM, "years_experience": 60, "skills": ["x" * 80] * 100}
    assert models.MatchRequest.model_validate(request(query=edge))


@pytest.mark.parametrize(
    "payload",
    [
        request(k=0),
        request(k=51),
        request(query={"kind": "other"}),
        request(direction="sideways"),
        request(system="Best"),
        request(extra=1),
        request(query={**FORM, "extra": 1}),
        request(query={**FORM, "skills": ["a"] * 101}),
        request(query={**FORM, "skills": ["x" * 81]}),
        request(query={**FORM, "years_experience": -1}),
        request(query={**FORM, "years_experience": 61}),
        request(query={"kind": "corpus", "id": ""}),
    ],
)
def test_match_request_rejects_invalid(payload: dict) -> None:
    with pytest.raises(ValidationError):
        models.MatchRequest.model_validate(payload)


def test_response_models_round_trip() -> None:
    evidence = models.Evidence(
        shared=[models.SharedSkill(skill="Python", cluster="Programming")],
        related=[
            models.RelatedGroup(cluster="Front-end", resume_skills=["Vue"], posting_skills=["R"])
        ],
        gaps=[models.GapSkill(skill="Kubernetes", cluster="DevOps")],
        level=models.LevelComparison(resume="Associate", posting="Associate", relation="same"),
        title_overlap=["dev"],
    )
    provenance = models.Provenance(
        alias_table_version="a", taxonomy_version="t", serialization_version="s",
        relabel_mode="lookup", demo=True,
        embedding=models.EmbeddingIdentity(model="m", backend="b", dim=4),
    )
    capability = models.Capability(available=False, reason="why")
    info = models.ExplainerInfo(kind="template")
    health = models.HealthResponse(
        mode="demo", artifacts_dir=None, counts={"resumes": 1, "postings": 1},
        systems=["Full"], levels=["Entry level"], versions=provenance,
        capabilities={"structure": capability}, explainer=info,
    )
    match = models.Match(rank=1, id="j-1", score=0.5, document=doc("posting"), evidence=evidence)
    response = models.MatchResponse(
        direction="resume_to_posting", system="Full", query_kind="corpus", k=10, k_effective=1,
        query=doc(), matches=[match], notes=[models.Note(code="no_skills", message="m")],
        provenance=provenance,
    )
    instances = [
        health,
        models.ExamplesResponse(items=[doc()]),
        response,
        models.Explanation(summary="s", bullets=["b"], evidence=evidence, explainer=info),
        models.ErrorEnvelope(error=models.ErrorBody(code="internal", message="m")),
        models.ExplainRequest(direction="resume_to_posting", query=doc(), match_id="j-1"),
    ]
    assert health.status == "ok"
    for instance in instances:
        cls = type(instance)
        assert cls.model_validate(instance.model_dump()) == instance
        assert cls.model_validate_json(instance.model_dump_json()) == instance


def test_error_codes_table() -> None:
    assert models.ERROR_CODES == {
        "invalid_request": 422, "empty_query": 400, "text_too_long": 400, "side_mismatch": 400,
        "not_found": 404, "embedding_mismatch": 409, "structuring_failed": 502,
        "embedding_failed": 502, "capability_unavailable": 503, "internal": 500,
    }
    with pytest.raises(ValidationError):
        models.ErrorEnvelope.model_validate({"error": {"code": "internal"}})
    with pytest.raises(ValidationError):
        models.ErrorEnvelope.model_validate({"error": {"message": "m"}})
    with pytest.raises(ValidationError):
        models.ErrorEnvelope.model_validate({})

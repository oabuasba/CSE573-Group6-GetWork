import pytest

from getwork.agent import tools
from getwork.agent.facts import FactBook
from getwork.agent.models import QUERY_DOC, ChatMatchRef
from getwork.webapp import demo, engine, models

pytestmark = pytest.mark.usefixtures("offline")


class RecordingEngine(demo.DemoEngine):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def get_document(self, doc_id: str) -> models.DocumentView:
        self.calls.append("get_document")
        return super().get_document(doc_id)

    def match(self, request: models.MatchRequest) -> engine.MatchResult:
        self.calls.append("match")
        return super().match(request)

    def health(self) -> models.EngineHealth:
        self.calls.append("health")
        return super().health()

    def examples(self, side: models.Side, limit: int) -> list[models.DocumentView]:
        self.calls.append("examples")
        return super().examples(side, limit)


def make_ctx(eng: demo.DemoEngine | None = None, *, ad_hoc: bool = False) -> tools.TurnContext:
    eng = eng or demo.DemoEngine()
    query = eng.get_document("r-1")
    if ad_hoc:
        query = query.model_copy(update={"id": None})
    refs = {
        "j-1": ChatMatchRef(id="j-1", rank=1, score=0.8333),
        "j-2": ChatMatchRef(id="j-2", rank=2, score=0.0),
    }
    docs = {QUERY_DOC: query, "j-1": eng.get_document("j-1")}
    return tools.TurnContext(
        engine=eng,
        direction="resume_to_posting",
        system="Full",
        query=query,
        focus_id="j-1",
        refs=refs,
        docs=docs,
        book=FactBook(),
    )


def test_document_skills_and_compare() -> None:
    ctx = make_ctx()
    created = tools.run_tool(ctx, "document_skills", {"doc_id": "j-2"})
    assert [f.kind for f in created] == ["document_skills"]
    assert created[0].text == (
        "Skills of j-2 (Java backend engineer, Mid-Senior level) by cluster: "
        "Back-end: Java, Spring; Data: SQL."
    )
    assert "j-2" in ctx.docs
    again = tools.run_tool(ctx, "document_skills", {"doc_id": "query"})
    assert again[0].doc_ids == ["query"]
    created = tools.run_tool(ctx, "compare_matches", {"id_a": "j-1", "id_b": "j-2"})
    assert [f.kind for f in created] == ["rank_score", "rank_score", "comparison"]
    assert created[0].text == "j-1 is ranked 1 with similarity 0.833."
    assert created[1].text == "j-2 is ranked 2 with similarity 0.000."
    assert created[2].text == (
        "j-1 vs j-2: shared with j-1 only: JavaScript, CSS; shared with j-2 only: none; "
        "gaps only for j-1: none; gaps only for j-2: Java, Spring, SQL."
    )
    assert created[2].doc_ids == ["j-1", "j-2"]
    assert list(ctx.book.facts) == ["f1", "f2", "f3", "f4", "f5"]


def test_cluster_lookup_and_top_matches() -> None:
    ctx = make_ctx()
    by_cluster = tools.run_tool(ctx, "cluster_lookup", {"name": "front-end"})[0]
    assert by_cluster.kind == "cluster_members" and by_cluster.cluster == "Front-end"
    assert by_cluster.text == "In Front-end: query: Vue, CSS, JavaScript; j-1: React, JavaScript, CSS."
    by_skill = tools.run_tool(ctx, "cluster_lookup", {"name": "react"})[0]
    assert by_skill.text == by_cluster.text
    created = tools.run_tool(ctx, "top_matches", {"doc_id": "j-1", "k": 99})
    kinds = [f.kind for f in created]
    assert kinds == ["rank_score", "document_skills"] * 5
    assert created[0].text.startswith("r-1 is ranked 1 with similarity 0.")
    assert "r-1" in ctx.docs
    default = tools.run_tool(make_ctx(), "top_matches", {"doc_id": "query"})
    assert len(default) == 6
    assert default[0].text.startswith("j-1 is ranked 1")
    with pytest.raises(tools.ToolError):
        tools.run_tool(make_ctx(ad_hoc=True), "top_matches", {"doc_id": "query"})


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("delete_everything", {}),
        ("document_skills", {}),
        ("document_skills", {"doc_id": 3}),
        ("document_skills", {"doc_id": "j-1", "extra": 1}),
        ("document_skills", {"doc_id": "j-3"}),
        ("compare_matches", {"id_a": "j-1"}),
        ("compare_matches", {"id_a": "j-1", "id_b": "j-1"}),
        ("compare_matches", {"id_a": "j-1", "id_b": "query"}),
        ("compare_matches", {"id_a": "j-1", "id_b": "j-3"}),
        ("cluster_lookup", {"name": "Kubernetes"}),
        ("cluster_lookup", {"name": ""}),
        ("top_matches", {"doc_id": "j-1", "k": "3"}),
        ("top_matches", {"doc_id": "j-1", "k": 0}),
        ("top_matches", {"doc_id": "j-1", "k": True}),
        ("top_matches", {"doc_id": "j-3"}),
    ],
)
def test_tool_errors(name: str, args: dict) -> None:
    ctx = make_ctx()
    before = dict(ctx.book.facts)
    with pytest.raises(tools.ToolError) as raised:
        tools.run_tool(ctx, name, args)
    assert str(raised.value)
    assert ctx.book.facts == before


def test_engine_not_found_becomes_tool_error() -> None:
    ctx = make_ctx()
    ctx.refs["j-404"] = ChatMatchRef(id="j-404", rank=3, score=0.1)
    with pytest.raises(tools.ToolError, match="j-404"):
        tools.run_tool(ctx, "document_skills", {"doc_id": "j-404"})


def test_tools_only_use_engine_reads() -> None:
    eng = RecordingEngine()
    ctx = make_ctx(eng)
    eng.calls.clear()
    tools.run_tool(ctx, "document_skills", {"doc_id": "j-2"})
    tools.run_tool(ctx, "compare_matches", {"id_a": "j-1", "id_b": "j-2"})
    tools.run_tool(ctx, "cluster_lookup", {"name": "SQL"})
    tools.run_tool(ctx, "top_matches", {"doc_id": "j-1", "k": 2})
    assert set(eng.calls) == {"get_document", "match"}
    assert set(tools.TOOL_NAMES) == set(tools.TOOL_DESCRIPTIONS)
    assert all(name in text for name, text in tools.TOOL_DESCRIPTIONS.items())

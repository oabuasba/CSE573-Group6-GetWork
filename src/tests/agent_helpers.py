"""Shared builders for the agent turn tests (not a test module)."""

import json
import time
from collections.abc import Callable
from typing import Any

from getwork.agent import facts, graph, grounding, models, state, tools, turn
from getwork.agent import vocab as vocab_loader
from getwork.agent.explainer import ChatService
from getwork.llm.fake import FakeLLMClient
from getwork.llm.retry import RetryConfig
from getwork.webapp import demo, explain

SECRET = "sk-secret-123"
VOCAB = grounding.make_vocabulary(
    ["Vue", "React", "CSS", "JavaScript", "Java", "Spring", "SQL", "Kubernetes"], ["Front-end", "Back-end", "Data"]
)


def make_ctx() -> tools.TurnContext:
    eng = demo.DemoEngine()
    query = eng.get_document("r-1").model_copy(update={"id": None})
    match = eng.get_document("j-1")
    book = facts.FactBook()
    evidence = explain.build_evidence("resume_to_posting", query, match)
    facts.evidence_facts(book, "resume_to_posting", query, match, evidence)
    refs = {
        "j-1": models.ChatMatchRef(id="j-1", rank=1, score=0.8333),
        "j-2": models.ChatMatchRef(id="j-2", rank=2, score=0.0),
    }
    return tools.TurnContext(
        engine=eng,
        direction="resume_to_posting",
        system="Full",
        query=query,
        focus_id="j-1",
        refs=refs,
        docs={models.QUERY_DOC: query, "j-1": match},
        book=book,
    )


def make_runner(
    script: list,
    *,
    clock: Callable[[], float] = time.monotonic,
    vocab: grounding.Vocabulary = VOCAB,
    **cfg: Any,
) -> tuple[turn.AgentRunner, FakeLLMClient]:
    fake = FakeLLMClient(script=script)
    config = state.AgentConfig(model="demo-model", backend="voyager", retry=RetryConfig(max_attempts=2), **cfg)
    runner = turn.AgentRunner(fake, config, vocab, secrets=[SECRET], sleep=lambda s: None, clock=clock)
    return runner, fake


def run(runner: turn.AgentRunner, max_tool_steps: int | None = None) -> models.AgentResult:
    started = runner.new_turn(
        ctx=make_ctx(), question="Why is j-1 first?", history=[], max_tool_steps=max_tool_steps
    )
    return turn.drive(runner, started)


def answer(text: str = "Both documents list CSS and JavaScript.") -> str:
    return text


def good_answer() -> str:
    return answer("Both documents list JavaScript and CSS.")


def tool_call(tool: str, **args: object) -> str:
    return json.dumps({"action": "tool", "tool": tool, "args": args})


def make_agent(
    script: list, *, vocab: grounding.Vocabulary | None = None, **cfg: Any
) -> tuple[ChatService, FakeLLMClient]:
    """Chat service over one scripted fake, the demo engine and the demo vocabulary."""
    runner, fake = make_runner(script, vocab=vocab or vocab_loader.demo_vocabulary(), **cfg)
    return ChatService(graph.build_graph(runner), runner, demo.DemoEngine()), fake


def chat_request(**overrides: Any) -> models.ChatRequest:
    eng = demo.DemoEngine()
    base = {
        "direction": "resume_to_posting",
        "query": eng.get_document("r-1").model_dump(),
        "focus_id": "j-1",
        "matches": [{"id": "j-1", "rank": 1, "score": 0.8333}, {"id": "j-2", "rank": 2, "score": 0.0}],
        "message": "Why is j-1 ranked above j-2?",
    }
    return models.ChatRequest.model_validate({**base, **overrides})


def outside_vocab() -> grounding.Vocabulary:
    """Demo vocabulary plus two skills that appear in no demo document (Terraform, Cluster)."""
    base = vocab_loader.demo_vocabulary()
    return grounding.make_vocabulary([*base.skills.values(), "Terraform", "Cluster"], base.clusters)


def make_client(script: list, *, vocab: grounding.Vocabulary | None = None):  # noqa: ANN201
    """TestClient over the demo engine with the rules explainer and /api/chat registered."""
    from fastapi.testclient import TestClient

    from getwork.agent.api import register_chat_routes
    from getwork.webapp import api

    chat, fake = make_agent(script, vocab=vocab)
    app = api.create_app(demo.DemoEngine(), explain.TemplateExplainer(), secrets=[SECRET])
    register_chat_routes(app, chat, secrets=[SECRET])
    return TestClient(app, raise_server_exceptions=False), fake


def chat_json(**overrides: Any) -> dict:
    return chat_request(**overrides).model_dump(mode="json")


def explain_json(match_id: str = "j-1") -> dict:
    query = demo.DemoEngine().get_document("r-1").model_dump(mode="json")
    return {"direction": "resume_to_posting", "query": query, "match_id": match_id}

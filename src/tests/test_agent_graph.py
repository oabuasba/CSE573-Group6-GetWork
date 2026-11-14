import json
import subprocess
import sys

import pytest
from agent_helpers import VOCAB, answer, make_ctx, make_runner, tool_call

from getwork.agent import graph, grounding, models, state, turn
from getwork.llm.client import FatalLLMError, TransientLLMError
from getwork.llm.fake import FakeLLMClient

pytestmark = pytest.mark.usefixtures("offline")

OUTSIDE = answer("Both list CSS and Kubernetes.")
CALL = tool_call("cluster_lookup", name="CSS")
# (script, runner options, max_tool_steps)
SCRIPTS = {
    "direct": ([answer()], {}, None),
    "outside_skill": ([OUTSIDE], {}, None),
    "empty_then_ok": ([""] * 2 + [answer()], {}, None),
    "tool_then_answer": ([tool_call("compare_matches", id_a="j-1", id_b="j-2"), answer()], {}, None),
    "tool_error": ([tool_call("compare_matches", id_a="j-1", id_b="j-9"), answer()], {}, None),
    "tool_limit": ([CALL, CALL, answer()], {"max_tool_steps": 1}, None),
    "no_tools_allowed": ([CALL, answer()], {}, 0),
    "endless_tools": ([CALL] * 10, {}, None),
    "fatal": ([FatalLLMError("no", status=401)], {}, None),
    "transient": ([TransientLLMError("down", status=503)] * 2, {}, None),
    "unparsable": ([""] * 3, {}, None),
}


def summary(result: models.AgentResult) -> tuple:
    return (
        result.status,
        result.reply,
        result.failure_reason,
        result.usage.llm_calls,
        result.ungrounded_skills,
    )


def test_graph_matches_driver_with_tool() -> None:
    script = SCRIPTS["tool_then_answer"][0]
    plain_runner, _ = make_runner(list(script))
    plain = turn.drive(plain_runner, plain_runner.new_turn(ctx=make_ctx(), question="Q", history=[]))
    runner, _ = make_runner(list(script))
    result = graph.run_agent(graph.build_graph(runner), runner, ctx=make_ctx(), question="Q", history=[])
    assert result.status == "ok" and summary(result) == summary(plain)
    assert [t.tool for t in result.trace] == ["compare_matches"] == [t.tool for t in plain.trace]


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_graph_matches_plain_driver(name: str) -> None:
    script, options, tool_steps = SCRIPTS[name]
    plain_runner, _ = make_runner(list(script), **options)
    plain = turn.drive(
        plain_runner,
        plain_runner.new_turn(ctx=make_ctx(), question="Q", history=[], max_tool_steps=tool_steps),
    )
    runner, _ = make_runner(list(script), **options)
    result = graph.run_agent(
        graph.build_graph(runner), runner, ctx=make_ctx(), question="Q", history=[], max_tool_steps=tool_steps
    )
    assert summary(result) == summary(plain)
    assert [t.tool for t in result.trace] == [t.tool for t in plain.trace]


def test_no_verify_step_anywhere() -> None:
    assert graph.NODE_NAMES == ("decide", "run_tool", "finish", "fail")
    assert not hasattr(turn.AgentRunner, "verify")
    assert not hasattr(grounding, "verify_claims")
    with pytest.raises(TypeError):
        state.AgentConfig(model="m", backend="b", max_verify_retries=1)  # type: ignore[call-arg]
    runner, _ = make_runner([answer()])
    assert "verify" not in graph.build_graph(runner).get_graph().nodes


def test_run_agent_result_identity() -> None:
    runner, _ = make_runner([answer()])
    result = graph.run_agent(graph.build_graph(runner), runner, ctx=make_ctx(), question="Q", history=[])
    assert (result.model, result.backend, result.prompt_version) == ("demo-model", "voyager", "agent_v3")
    assert result.status == "ok" and json.dumps(result.as_record())


def test_recursion_limit_becomes_failure() -> None:
    fake = FakeLLMClient(handler=lambda request: CALL)
    cfg = state.AgentConfig(model="m", backend="b", max_llm_calls=500, max_tool_steps=500)
    runner = turn.AgentRunner(fake, cfg, VOCAB, sleep=lambda s: None)
    result = graph.run_agent(graph.build_graph(runner), runner, ctx=make_ctx(), question="Q", history=[])
    assert result.status == "failed" and result.failure_reason == "budget_exhausted"
    assert result.reply == "" and 0 < len(fake.calls) < 500


def test_unexpected_exception_propagates() -> None:
    runner, _ = make_runner([RuntimeError("bug in a node")])
    with pytest.raises(RuntimeError, match="bug in a node"):
        graph.run_agent(graph.build_graph(runner), runner, ctx=make_ctx(), question="Q", history=[])


def run_python(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)


def test_langgraph_imported_lazily() -> None:
    code = (
        "import sys\n"
        "import getwork.agent.turn, getwork.agent.explainer, getwork.agent.api, getwork.agent.graph\n"
        "import getwork.webapp.api\n"
        "sys.exit(1 if any(m == 'langgraph' or m.startswith('langgraph.') for m in sys.modules) else 0)\n"
    )
    done = run_python(code)
    assert done.returncode == 0, done.stderr


def test_missing_langgraph_message() -> None:
    code = (
        "import sys\n"
        "sys.modules['langgraph'] = None\n"
        "from getwork.agent import graph\n"
        "try:\n"
        "    graph.build_graph(None)\n"
        "except RuntimeError as e:\n"
        "    print(e)\n"
        "    sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    done = run_python(code)
    assert done.returncode == 0, done.stderr
    assert "uv sync --extra agent" in done.stdout

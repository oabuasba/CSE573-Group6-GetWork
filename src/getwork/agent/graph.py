"""LangGraph wiring of the AgentRunner nodes; langgraph is imported only when a graph is built."""

from collections.abc import Sequence
from typing import Any, TypedDict

from getwork.agent.models import AgentResult, ChatMessage
from getwork.agent.state import Turn
from getwork.agent.tools import TurnContext
from getwork.agent.turn import AgentRunner

RECURSION_LIMIT = 25
MISSING_LANGGRAPH = "langgraph is not installed; run: uv sync --extra agent"
NODE_NAMES = ("decide", "run_tool", "finish", "fail")
BRANCHING_NODES = ("decide", "run_tool")


class GraphState(TypedDict):
    turn: Turn
    route: str


def _make_node(method: Any):  # noqa: ANN202 - a plain state-update function for StateGraph
    def node(state: GraphState) -> dict[str, Any]:
        turn = method(state["turn"])
        return {"turn": turn, "route": turn.route}

    return node


def _read_route(state: GraphState) -> str:
    return state["route"]


def build_graph(runner: AgentRunner):  # noqa: ANN201 - the compiled graph type belongs to langgraph
    """Compiles the StateGraph described in "Graph, state and flow": state keys turn (Turn) and route (str);
    nodes call the runner methods; conditional edges read route; recursion_limit 25 at invoke. ImportError from the
    missing package is re-raised as RuntimeError("langgraph is not installed; run: uv sync --extra agent")."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as e:
        raise RuntimeError(MISSING_LANGGRAPH) from e
    graph = StateGraph(GraphState)
    for name in NODE_NAMES:
        graph.add_node(name, _make_node(getattr(runner, name)))
    graph.set_entry_point("decide")
    targets = {name: name for name in NODE_NAMES}
    for name in BRANCHING_NODES:
        graph.add_conditional_edges(name, _read_route, targets)
    graph.add_edge("finish", END)
    graph.add_edge("fail", END)
    return graph.compile()


def run_agent(
    graph: Any,
    runner: AgentRunner,
    *,
    ctx: TurnContext,
    question: str,
    history: Sequence[ChatMessage],
    max_tool_steps: int | None = None,
) -> AgentResult:
    """new_turn, graph.invoke, return turn.result. A GraphRecursionError becomes a failed result with reason
    budget_exhausted; any other exception propagates (it is a bug, and the web-app turns it into the generic 500)."""
    from langgraph.errors import GraphRecursionError

    turn = runner.new_turn(ctx=ctx, question=question, history=history, max_tool_steps=max_tool_steps)
    try:
        final = graph.invoke({"turn": turn, "route": turn.route}, {"recursion_limit": RECURSION_LIMIT})
    except GraphRecursionError:
        turn.failure_reason = "budget_exhausted"
        runner.fail(turn)
        final = {"turn": turn}
    result = final["turn"].result
    if result is None:
        raise RuntimeError("the graph ended without a result")
    return result

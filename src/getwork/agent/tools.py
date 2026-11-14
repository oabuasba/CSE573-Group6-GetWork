"""The four read-only tools the agent may call; they only use Engine.get_document and Engine.match."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from getwork.agent import facts as fact_builders
from getwork.agent.models import QUERY_DOC, ChatMatchRef, Fact
from getwork.alias.normalize import normalize_key
from getwork.webapp.engine import Engine, EngineError
from getwork.webapp.explain import build_evidence
from getwork.webapp.models import CorpusQuery, Direction, DocumentView, MatchRequest, SystemId

TOOL_NAMES: tuple[str, ...] = ("document_skills", "compare_matches", "cluster_lookup", "top_matches")
TOOL_DESCRIPTIONS: dict[str, str] = {
    "document_skills": 'document_skills(doc_id): skills of one document by cluster; doc_id is "query" or a result id.',
    "compare_matches": "compare_matches(id_a, id_b): ranks and what sets two results apart; ids from the result list.",
    "cluster_lookup": "cluster_lookup(name): which skills in scope sit in a cluster; name is a skill or a cluster.",
    "top_matches": "top_matches(doc_id, k): best matches for another document in scope; k from 1 to 5, default 3.",
}
MAX_TOP_K = 5
DEFAULT_TOP_K = 3


class ToolError(Exception):
    """A tool request the agent can correct; the message goes back to the model."""


@dataclass
class TurnContext:
    engine: Engine
    direction: Direction
    system: SystemId
    query: DocumentView
    focus_id: str
    refs: dict[str, ChatMatchRef]  # the result list by id
    docs: dict[str, DocumentView]  # scope: QUERY_DOC, the focus and every document loaded this turn
    book: fact_builders.FactBook


def _text_arg(args: Mapping[str, object], name: str) -> str:
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"argument {name} must be a non-empty string.")
    return value.strip()


def _check_arg_names(args: Mapping[str, object], allowed: Sequence[str]) -> None:
    extra = sorted(set(args) - set(allowed))
    if extra:
        raise ToolError(f"unknown argument {extra[0]}; this tool takes {', '.join(allowed)}.")


def _load(ctx: TurnContext, doc_id: str) -> DocumentView:
    if doc_id in ctx.docs:
        return ctx.docs[doc_id]
    try:
        doc = ctx.engine.get_document(doc_id)
    except EngineError as e:
        raise ToolError(str(e)) from e
    ctx.docs[doc_id] = doc
    return doc


def _require_known(ctx: TurnContext, doc_id: str) -> None:
    if doc_id not in ctx.refs and doc_id not in ctx.docs:
        raise ToolError(f"{doc_id} is not part of this conversation.")


def _document_skills(ctx: TurnContext, args: Mapping[str, object]) -> list[Fact]:
    _check_arg_names(args, ["doc_id"])
    doc_id = _text_arg(args, "doc_id")
    _require_known(ctx, doc_id)
    doc = _load(ctx, doc_id)
    if doc_id == QUERY_DOC:
        doc = doc.model_copy(update={"id": None})  # the user's document is always called "query" in facts
    return [fact_builders.document_fact(ctx.book, doc)]


def _names(skills: Sequence[str], exclude: set[str]) -> list[str]:
    return [s for s in skills if normalize_key(s) not in exclude]


def _compare_matches(ctx: TurnContext, args: Mapping[str, object]) -> list[Fact]:
    _check_arg_names(args, ["id_a", "id_b"])
    id_a, id_b = _text_arg(args, "id_a"), _text_arg(args, "id_b")
    for doc_id in (id_a, id_b):
        if doc_id not in ctx.refs:
            raise ToolError(f"{doc_id} is not in the result list.")
    if id_a == id_b:
        raise ToolError("id_a and id_b must be different results.")
    evidence = {i: build_evidence(ctx.direction, ctx.query, _load(ctx, i)) for i in (id_a, id_b)}
    shared = {i: [s.skill for s in evidence[i].shared] for i in evidence}
    gaps = {i: [s.skill for s in evidence[i].gaps] for i in evidence}
    keys = {i: {normalize_key(s) for s in shared[i]} for i in evidence}
    gap_keys = {i: {normalize_key(s) for s in gaps[i]} for i in evidence}
    parts = {
        f"shared with {id_a} only": _names(shared[id_a], keys[id_b]),
        f"shared with {id_b} only": _names(shared[id_b], keys[id_a]),
        f"gaps only for {id_a}": _names(gaps[id_a], gap_keys[id_b]),
        f"gaps only for {id_b}": _names(gaps[id_b], gap_keys[id_a]),
    }
    created = [
        fact_builders.rank_fact(ctx.book, i, rank=ctx.refs[i].rank, score=ctx.refs[i].score, system=ctx.system)
        for i in (id_a, id_b)
    ]
    text = f"{id_a} vs {id_b}: " + "; ".join(f"{k}: {fact_builders.join_names(v)}" for k, v in parts.items()) + "."
    skills = [s for names in parts.values() for s in names]
    created.append(ctx.book.add("comparison", doc_ids=[id_a, id_b], skills=skills, text=text))
    return created


def _find_cluster(ctx: TurnContext, name: str) -> str:
    key = normalize_key(name)
    for doc in ctx.docs.values():
        for item in doc.skills:
            if key in (normalize_key(item.cluster), normalize_key(item.skill)):
                return item.cluster
    raise ToolError(f"{name} is not a skill or cluster in the documents of this conversation.")


def _cluster_lookup(ctx: TurnContext, args: Mapping[str, object]) -> list[Fact]:
    _check_arg_names(args, ["name"])
    cluster = _find_cluster(ctx, _text_arg(args, "name"))
    members = {ref: [s.skill for s in doc.skills if s.cluster == cluster] for ref, doc in ctx.docs.items()}
    members = {ref: skills for ref, skills in members.items() if skills}
    text = f"In {cluster}: " + "; ".join(f"{ref}: {', '.join(skills)}" for ref, skills in members.items()) + "."
    all_skills = [s for skills in members.values() for s in skills]
    fact = ctx.book.add("cluster_members", doc_ids=list(members), skills=all_skills, text=text, cluster=cluster)
    return [fact]


def _top_k(args: Mapping[str, object]) -> int:
    k = args.get("k", DEFAULT_TOP_K)
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ToolError("argument k must be a whole number from 1 to 5.")
    return min(k, MAX_TOP_K)


def _top_matches(ctx: TurnContext, args: Mapping[str, object]) -> list[Fact]:
    _check_arg_names(args, ["doc_id", "k"])
    doc_id, k = _text_arg(args, "doc_id"), _top_k(args)
    _require_known(ctx, doc_id)
    if doc_id == QUERY_DOC:
        if ctx.query.id is None:
            raise ToolError("the query document has no id to search with; use a result id instead.")
        doc_id = ctx.query.id
    doc = _load(ctx, doc_id)
    direction: Direction = "resume_to_posting" if doc.side == "resume" else "posting_to_resume"
    request = MatchRequest(direction=direction, query=CorpusQuery(kind="corpus", id=doc_id), k=k, system=ctx.system)
    try:
        result = ctx.engine.match(request)
    except EngineError as e:
        raise ToolError(str(e)) from e
    created: list[Fact] = []
    for rank, (found, score) in enumerate(result.matches, start=1):
        ref = fact_builders.doc_ref(found)
        ctx.docs[ref] = found
        created.append(fact_builders.rank_fact(ctx.book, ref, rank=rank, score=score, system=ctx.system))
        created.append(fact_builders.document_fact(ctx.book, found))
    return created


_TOOLS: dict[str, Callable[[TurnContext, Mapping[str, object]], list[Fact]]] = {
    "document_skills": _document_skills,
    "compare_matches": _compare_matches,
    "cluster_lookup": _cluster_lookup,
    "top_matches": _top_matches,
}


def run_tool(ctx: TurnContext, name: str, args: Mapping[str, object]) -> list[Fact]:
    """ToolError for an unknown tool, missing or wrongly typed arguments (doc_id/id_a/id_b/name strings, k int 1 to
    5 default 3), an id outside ctx.refs/ctx.docs, an unknown skill or cluster, or an EngineError from the engine
    (message kept). Adds loaded documents to ctx.docs and facts to ctx.book; returns the new facts."""
    tool = _TOOLS.get(name)
    if tool is None:
        raise ToolError(f"unknown tool {name}; use one of {', '.join(TOOL_NAMES)}.")
    return tool(ctx, args)

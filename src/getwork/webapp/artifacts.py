"""ArtifactEngine: serves real artifacts from a run folder (ADR-12, ADR-15)."""

import logging
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from getwork import fileio, paths
from getwork.alias.apply import canonicalize_record
from getwork.alias.cli import TABLE_FILE
from getwork.alias.table import AliasTable, load_table
from getwork.docembed import documents, levels, vectors
from getwork.docembed.cli_build import TAXONOMY_FILE
from getwork.embed.client import EmbeddingClient, EmbeddingError, EmbeddingMismatchError
from getwork.embed.store import VectorSet, check_same_embedding
from getwork.llm.client import FatalLLMError, LLMClient, LLMError
from getwork.llm.retry import Throttle
from getwork.llm.settings import ConfigError
from getwork.relabel.lookup import SkillAssignment, incomplete_reason, known_fraction, split_known
from getwork.relabel.records import build_relabel_record
from getwork.retrieval import inputs as retrieval_inputs
from getwork.retrieval.b1plus import DEFAULT_WEIGHTS, B1PlusWeights, b1plus_score
from getwork.retrieval.topk import matrix_block, topk_from_blocks
from getwork.structure.extract import ExtractConfig, SourceDoc, structure_one
from getwork.structure.schema import RECORD_KEYS, SCHEMA_VERSION
from getwork.taxonomy.model import OTHER, Taxonomy, load_taxonomy
from getwork.webapp import models
from getwork.webapp.engine import EngineError, EngineLoadError, MatchResult

logger = logging.getLogger(__name__)

REQUIRED_ARTIFACT_DIRS: tuple[str, ...] = ("alias", "documents", "taxonomy")
RUN_FOLDER = re.compile(r"v(\d+)")
RUN_HINT = (
    "pass the run folder itself (for example artifacts_trial/v5) or build the missing stages "
    "(alias-table build, taxonomy build, doc-embed build)"
)
QUERY_ID = "q-1"
STRUCTURING_REJECTED = "The model service rejected the request"
SOURCE_OF: dict[str, str] = {"resume": "resumes", "posting": "postings"}
QUERY_SIDE: dict[str, models.Side] = {"resume_to_posting": "resume", "posting_to_resume": "posting"}
OTHER_SIDE: dict[models.Side, models.Side] = {"resume": "posting", "posting": "resume"}
NOTE_MESSAGES: dict[str, str] = {
    "no_skills": "No skills were found, so results rely on the title and level only.",
    "sparse_query": "Few of the skills are known, so results rely more on the title.",
    "unknown_skills": "None of the skills are in the skill taxonomy, so none earn related credit.",
    "relabel_mode_differs": (
        "The stored documents were labeled with a language model; this query used the "
        "taxonomy lookup only."
    ),
}


def _missing_dirs(root: Path) -> list[str]:
    return [name for name in REQUIRED_ARTIFACT_DIRS if not (root / name).is_dir()]


def resolve_artifacts_root(path: Path) -> Path:
    """The run folder to load: path itself when complete, else its highest complete vN subfolder."""
    if not path.is_dir():
        raise EngineLoadError(f"artifacts directory not found: {path} ({RUN_HINT})")
    missing = _missing_dirs(path)
    if not missing:
        return path
    runs = sorted(
        ((int(m[1]), child) for child in path.iterdir()
         if child.is_dir() and (m := RUN_FOLDER.fullmatch(child.name))),
        key=lambda run: run[0], reverse=True,
    )
    skipped = []
    for _, child in runs:
        if not _missing_dirs(child):
            logger.info("artifacts run %s chosen in %s; skipped incomplete %s", child.name, path, skipped)
            return child
        skipped.append(child.name)
    found = "; ".join(f"{child.name} lacks {_missing_dirs(child)}" for _, child in runs)
    raise EngineLoadError(
        f"{path} is not a complete artifacts run: missing folders {missing}"
        + (f" and no complete vN subfolder ({found})" if runs else " and has no vN subfolders")
        + f" ({RUN_HINT})"
    )


@dataclass(frozen=True)
class StructureResources:
    client: LLMClient
    cfg: ExtractConfig
    secrets: Sequence[str] = ()


@dataclass(frozen=True)
class EmbedResources:
    factory: Callable[[], EmbeddingClient]
    describe: str


@dataclass
class _Embedder:
    """Lazy, cached, lock-protected embedding client (CC-13)."""

    resources: EmbedResources
    lock: threading.Lock = field(default_factory=threading.Lock)
    client: EmbeddingClient | None = None


def _view_of(doc: Mapping[str, object], *, anonymous: bool = False) -> models.DocumentView:
    return models.DocumentView(
        id=None if anonymous else doc["id"],
        side="resume" if doc["source"] == "resumes" else "posting",
        title=doc["title"],
        level=doc["level"],
        level_source=doc["level_source"],
        years_experience=doc["years_experience"],
        skills=[models.DocSkill(**sc) for sc in doc["skill_clusters"]],
        incomplete_reason=None,
    )


def _assign_skills(skills: Sequence[str], tax: Taxonomy) -> list[SkillAssignment]:
    known_iter = iter(split_known(skills=skills, tax=tax)[0])
    return [
        next(known_iter) if tax.is_known(s) else SkillAssignment(s, OTHER, "default")
        for s in skills
    ]


def _notes_for(
    assignments: Sequence[SkillAssignment], reason: str | None, corpus_mode: str | None
) -> list[models.Note]:
    codes = []
    if not assignments:
        codes.append("no_skills")
    elif reason is not None:
        codes.append("sparse_query")
    if assignments and all(a.cluster == OTHER for a in assignments):
        codes.append("unknown_skills")
    if corpus_mode == "llm":
        codes.append("relabel_mode_differs")
    return [models.Note(code=code, message=NOTE_MESSAGES[code]) for code in codes]


def _unavailable(what: str) -> EngineError:
    return EngineError("capability_unavailable", f"{what} is not available on this server.")


class ArtifactEngine:
    def __init__(
        self, *, root: Path, data: retrieval_inputs.RetrievalInputs, table: AliasTable,
        taxonomy: Taxonomy, p10: Mapping[str, int], systems: Sequence[str],
        weights: B1PlusWeights, structure: StructureResources | None, structure_reason: str | None,
        embed: EmbedResources | None, embed_reason: str | None, max_query_chars: int,
    ) -> None:
        self._root = root
        self._data = data
        self._table = table
        self._tax = taxonomy
        self._p10 = dict(p10)
        self._systems = [s for s in retrieval_inputs.SYSTEM_IDS if s in systems]
        self._weights = weights
        self._structure = structure
        self._structure_reason = structure_reason
        self._embedder = _Embedder(embed) if embed else None
        self._embed_reason = embed_reason
        self._max_query_chars = max_query_chars
        self._tables = {"resume": data.resumes, "posting": data.postings}
        self._docs = {side: list(table_.values()) for side, table_ in self._tables.items()}
        self._positions = {
            key: {id_: i for i, id_ in enumerate(vset.ids)} for key, vset in data.vectors.items()
        }
        self._provenance = self._build_provenance()

    @classmethod
    def load(
        cls, root: Path, *, systems: Sequence[str] = retrieval_inputs.SYSTEM_IDS,
        weights: B1PlusWeights = DEFAULT_WEIGHTS,
        structure: StructureResources | None = None, structure_reason: str | None = None,
        embed: EmbedResources | None = None, embed_reason: str | None = None,
        max_query_chars: int = models.DEFAULT_MAX_QUERY_CHARS,
    ) -> "ArtifactEngine":
        """root is any directory; it is resolved with resolve_artifacts_root first."""
        root = resolve_artifacts_root(root)
        try:
            data, table, taxonomy = cls._read(root=root, systems=systems)
            p10 = _read_p10s(root)
        except (
            FileNotFoundError, ValueError, documents.DocEmbedMismatchError, EmbeddingError,
        ) as exc:
            raise EngineLoadError(str(exc)) from exc
        _check_chain(table=table, taxonomy=taxonomy, summaries=data.summaries)
        return cls(
            root=root, data=data, table=table, taxonomy=taxonomy, p10=p10,
            systems=systems, weights=weights, structure=structure,
            structure_reason=structure_reason, embed=embed, embed_reason=embed_reason,
            max_query_chars=max_query_chars,
        )

    @staticmethod
    def _read(
        *, root: Path, systems: Sequence[str]
    ) -> tuple[retrieval_inputs.RetrievalInputs, AliasTable, Taxonomy]:
        alias_path = paths.alias_dir(root) / TABLE_FILE
        if not alias_path.exists():
            raise FileNotFoundError(f"alias table not found: {alias_path} (run alias-table build)")
        table = load_table(alias_path)
        taxonomy = load_taxonomy(paths.taxonomy_dir(root) / TAXONOMY_FILE)
        return retrieval_inputs.load_inputs(root, systems), table, taxonomy

    def _build_provenance(self) -> models.Provenance:
        summary = self._data.summaries["resumes"]
        meta = next((v.meta for (src, _), v in self._data.vectors.items() if src == "resumes"), None)
        return models.Provenance(
            alias_table_version=self._table.version,
            taxonomy_version=self._tax.version,
            serialization_version=summary.get("serialization_version"),
            relabel_mode=summary.get("relabel_mode"),
            embedding=models.EmbeddingIdentity(**{
                k: meta["embedding"][k] for k in ("model", "backend", "dim")
            }) if meta else None,
        )

    def health(self) -> models.EngineHealth:
        structure = self._structure
        embed = self._embedder.resources if self._embedder else None
        return models.EngineHealth(
            mode="artifacts", artifacts_dir=str(self._root),
            counts={"resumes": len(self._data.resumes), "postings": len(self._data.postings)},
            systems=list(self._systems),
            levels=[*levels.LEVELS, levels.NOT_SPECIFIED],
            versions=self._provenance,
            capabilities={
                "structure": models.Capability(
                    available=True, detail=f"{structure.cfg.backend}, {structure.cfg.model}"
                ) if structure else models.Capability(
                    available=False, reason=self._structure_reason or "No language model configured."
                ),
                "embed": models.Capability(available=True, detail=embed.describe)
                if embed else models.Capability(
                    available=False, reason=self._embed_reason or "No embedding model configured."
                ),
            },
        )

    def examples(self, side: models.Side, limit: int) -> list[models.DocumentView]:
        return [_view_of(doc) for doc in self._docs[side][:limit]]

    def get_document(self, doc_id: str) -> models.DocumentView:
        for table in self._tables.values():
            if doc_id in table:
                return _view_of(table[doc_id])
        raise EngineError("not_found", f"No document with id {doc_id!r}")

    def match(self, request: models.MatchRequest) -> MatchResult:
        kind = request.query.kind
        try:
            result = self._match(request)
        except EngineError as exc:
            logger.info("match kind=%s outcome=%s", kind, exc.code)
            raise
        logger.info("match kind=%s outcome=ok", kind)
        return result

    def _match(self, request: models.MatchRequest) -> MatchResult:
        system, side = request.system, QUERY_SIDE[request.direction]
        if system not in self._systems:
            raise _unavailable(f"The {system} system")
        query = request.query
        if isinstance(query, models.CorpusQuery):
            doc, notes = self._corpus_document(query.id, side), []
            vector = self._corpus_vector(side=side, system=system, doc_id=doc["id"])
        else:
            self._require_embed(system)
            record = (
                self._record_from_form(query, side) if isinstance(query, models.FormQuery)
                else self._record_from_text(query, side)
            )
            doc, notes = self._label_and_build(record=record, side=side, listed=_listed(query))
            vector = self._embed_one(doc, system)
        ranked = self._rank(request.direction, system, doc, vector, request.k)
        other = self._tables[OTHER_SIDE[side]]
        return MatchResult(
            query=_view_of(doc, anonymous=query.kind != "corpus"),
            matches=[(_view_of(other[id_]), score) for id_, score in ranked],
            notes=notes, provenance=self._provenance, k_effective=len(ranked),
        )

    def _corpus_document(self, doc_id: str, side: models.Side) -> dict:
        if doc_id in self._tables[side]:
            return self._tables[side][doc_id]
        code = "side_mismatch" if doc_id in self._tables[OTHER_SIDE[side]] else "not_found"
        message = (
            f"Document {doc_id!r} is on the wrong side for this search."
            if code == "side_mismatch" else f"No document with id {doc_id!r}"
        )
        raise EngineError(code, message, field="query.id")

    def _corpus_vector(self, *, side: models.Side, system: str, doc_id: str) -> np.ndarray | None:
        if system == "B1+":
            return None
        key = (SOURCE_OF[side], system)
        return self._data.vectors[key].vectors[self._positions[key][doc_id]]

    def _rank(
        self, direction: str, system: str, query_row: Mapping[str, object],
        query_vector: np.ndarray | None, k: int,
    ) -> list[tuple[str, float]]:
        other = OTHER_SIDE[QUERY_SIDE[direction]]
        docs = self._docs[other]
        if system == "B1+":
            is_resume_query = direction == "resume_to_posting"
            scores = np.array([
                b1plus_score(query_row, d, self._weights) if is_resume_query
                else b1plus_score(d, query_row, self._weights)
                for d in docs
            ], dtype=np.float32)
            def block(start: int, stop: int) -> np.ndarray:
                return scores[None, :]
        else:
            matrix = self._data.vectors[(SOURCE_OF[other], system)].vectors
            block = matrix_block(query_vector[None, :], matrix)
        indices, top = topk_from_blocks(block, 1, len(docs), k)
        return [(docs[i]["id"], float(s)) for i, s in zip(indices[0], top[0], strict=True)]

    def _require_embed(self, system: str) -> None:
        if system != "B1+" and self._embedder is None:
            raise EngineError(
                "capability_unavailable",
                self._embed_reason or "No embedding model configured.",
                hint="Use the keywords-only system, or start the server with an embedding model.",
            )

    def _record_from_form(self, form: models.FormQuery, side: models.Side) -> dict:
        skills = [s.strip() for s in form.skills if s.strip()]
        title = form.title.strip()
        if not title and not skills:
            raise EngineError("empty_query", "Add a title or some skills first.", field="query")
        record = _blank_record(side=side, title=title or None, skills=skills)
        record["years_experience"] = form.years_experience
        return canonicalize_record(record, self._table)

    def _record_from_text(self, query: models.TextQuery, side: models.Side) -> dict:
        if self._structure is None:
            raise EngineError(
                "capability_unavailable",
                self._structure_reason or "No language model configured.",
                hint="Use the form or pick an example instead.",
            )
        text = query.text.strip()
        if not text:
            raise EngineError("empty_query", "Add some text first.", field="query.text")
        if len(text) > self._max_query_chars:
            raise EngineError(
                "text_too_long",
                f"That text is longer than the limit of {self._max_query_chars} characters.",
                field="query.text",
            )
        title = query.title if side == "posting" else None
        source_doc = SourceDoc(QUERY_ID, SOURCE_OF[side], title, text)
        res = self._structure
        try:
            outcome = structure_one(res.client, res.cfg, Throttle(), source_doc, res.secrets)
        except FatalLLMError:
            raise EngineError("structuring_failed", STRUCTURING_REJECTED) from None
        if outcome.failure is not None:
            raise EngineError("structuring_failed", outcome.failure.message)
        return canonicalize_record(outcome.record, self._table)

    def _label_and_build(
        self, *, record: Mapping[str, object], side: models.Side, listed: str | None
    ) -> tuple[dict, list[models.Note]]:
        source = SOURCE_OF[side]
        skills = record["skills"]
        assignments = _assign_skills(skills, self._tax)
        reason = incomplete_reason(
            n_skills=len(skills), known=known_fraction(skills=skills, tax=self._tax),
            p10=self._p10[source],
        )
        relabeled = build_relabel_record(record, assignments, [], reason, False, self._tax, None)
        summary = self._data.summaries[source]
        doc = documents.build_document(
            relabeled, listed, self._tax, tuple(summary["level_bands"]), summary["b2_order"]
        )
        notes = _notes_for(assignments, reason, self._data.summaries["resumes"].get("relabel_mode"))
        return doc, notes

    def _embed_one(self, doc: Mapping[str, object], system: str) -> np.ndarray | None:
        if system == "B1+":
            return None
        text = doc[vectors.text_field(system)]
        embedder = self._embedder
        with embedder.lock:
            client = self._client_of(embedder)
            self._check_identity(client, system)
            try:
                return client.embed([text])[0]
            except (EmbeddingError, LLMError) as exc:
                raise EngineError(
                    "embedding_failed", f"The embedding step failed ({type(exc).__name__})."
                ) from None

    @staticmethod
    def _client_of(embedder: _Embedder) -> EmbeddingClient:
        if embedder.client is None:
            try:
                embedder.client = embedder.resources.factory()
            except Exception as exc:  # any construction failure disables only this capability
                # ConfigError text names variables, never values; other messages may not be safe.
                detail = str(exc) if isinstance(exc, ConfigError) else type(exc).__name__
                raise EngineError(
                    "capability_unavailable",
                    f"The embedding model could not be started ({detail}).",
                ) from None
        return embedder.client

    def _check_identity(self, client: EmbeddingClient, system: str) -> None:
        stored: VectorSet = self._data.vectors[("resumes", system)]
        try:
            check_same_embedding(a={"embedding": client.info.as_dict()}, b=stored.meta)
        except EmbeddingMismatchError as exc:
            raise EngineError(
                "embedding_mismatch", str(exc),
                hint="Start the server with the embedding model the documents were embedded with.",
            ) from None


def _listed(query: models.Query) -> str | None:
    return query.level if isinstance(query, models.FormQuery) else None


def _blank_record(*, side: models.Side, title: str | None, skills: list[str]) -> dict:
    record = dict.fromkeys(RECORD_KEYS)
    record.update(
        id=QUERY_ID, source=SOURCE_OF[side], title=title, skills=skills, experience=[],
        education=[], model="form", prompt_version="form_v1", schema_version=SCHEMA_VERSION,
        backend="form",
    )
    return record


def _read_p10s(root: Path) -> dict[str, int]:
    p10 = {}
    for source in paths.SOURCES:
        path = paths.relabeled_dir(root, source) / f"{source}_relabeled_summary.json"
        p10[source] = int(fileio.read_json(path).get("p10_threshold", 0)) if path.exists() else 0
    return p10


def _check_chain(*, table: AliasTable, taxonomy: Taxonomy, summaries: Mapping[str, dict]) -> None:
    if table.version != taxonomy.alias_table_version:
        raise EngineLoadError(
            f"alias table version {table.version} differs from the taxonomy's "
            f"alias_table_version {taxonomy.alias_table_version}; rebuild with alias-table build "
            f"and taxonomy build"
        )
    docs_version = summaries["resumes"].get("taxonomy_version")
    if taxonomy.version != docs_version:
        raise EngineLoadError(
            f"taxonomy version {taxonomy.version} differs from the documents' taxonomy_version "
            f"{docs_version}; run doc-embed build"
        )

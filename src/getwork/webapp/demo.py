"""Built-in demo data and DemoEngine: no artifacts, no key, no GPU (ADR-15)."""

from collections.abc import Collection

from getwork.alias import normalize
from getwork.docembed import levels
from getwork.taxonomy import model as taxonomy_model
from getwork.webapp import engine, models

OTHER_CLUSTER = taxonomy_model.OTHER
DEMO_VERSION = "demo"
SYSTEMS: list[models.SystemId] = ["B1+", "B2", "Full"]
FEW_SKILLS = 3
SPARSE_MESSAGE = "Your document lists few skills, so results rely more on the title."
NO_SKILLS_MESSAGE = "No skills were found, so results rely on the title and level only."
UNKNOWN_MESSAGE = "None of the skills are in the demo skill list, so none earn related credit."

DEMO_RESUMES: list[dict] = [
    {"id": "r-1", "title": "Front-end developer", "level": "Associate", "years_experience": 3.0,
     "skills": [("Vue", "Front-end"), ("CSS", "Front-end"), ("JavaScript", "Front-end")]},
    {"id": "r-2", "title": "Java developer", "level": "Mid-Senior level", "years_experience": 6.0,
     "skills": [("Java", "Back-end"), ("Spring", "Back-end"), ("SQL", "Data"),
                ("Docker", "DevOps")]},
    {"id": "r-3", "title": "Data analyst", "level": "Entry level", "years_experience": 1.0,
     "skills": [("Python", "Data"), ("SQL", "Data"), ("Tableau", "Data"),
                ("Excel", OTHER_CLUSTER)]},
    {"id": "r-4", "title": "DevOps engineer", "level": "Mid-Senior level",
     "years_experience": 7.0,
     "skills": [("Kubernetes", "DevOps"), ("Docker", "DevOps"), ("Terraform", "DevOps"),
                ("Python", "Data")]},
    {"id": "r-5", "title": "Mobile developer", "level": "Associate", "years_experience": 2.5,
     "skills": [("Swift", "Mobile"), ("Kotlin", "Mobile"), ("Git", OTHER_CLUSTER)]},
    {"id": "r-6", "title": "Project coordinator", "level": "Entry level",
     "years_experience": 0.5,
     "skills": [("Communication", OTHER_CLUSTER), ("Excel", OTHER_CLUSTER),
                ("Scheduling", OTHER_CLUSTER)]},
]

DEMO_POSTINGS: list[dict] = [
    {"id": "j-1", "title": "React developer", "level": "Associate", "years_experience": None,
     "skills": [("React", "Front-end"), ("JavaScript", "Front-end"), ("CSS", "Front-end")]},
    {"id": "j-2", "title": "Java backend engineer", "level": "Mid-Senior level",
     "years_experience": None,
     "skills": [("Java", "Back-end"), ("Spring", "Back-end"), ("SQL", "Data")]},
    {"id": "j-3", "title": "Data analyst", "level": "Entry level", "years_experience": None,
     "skills": [("SQL", "Data"), ("Tableau", "Data"), ("Excel", OTHER_CLUSTER),
                ("Reporting", OTHER_CLUSTER)]},
    {"id": "j-4", "title": "Junior DevOps engineer", "level": "Entry level",
     "years_experience": None,
     "skills": [("Kubernetes", "DevOps"), ("Docker", "DevOps"), ("Terraform", "DevOps")]},
    {"id": "j-5", "title": "Android developer", "level": "Associate", "years_experience": None,
     "skills": [("Kotlin", "Mobile"), ("Java", "Back-end"), ("Git", OTHER_CLUSTER)]},
    {"id": "j-6", "title": "Operations assistant", "level": "Entry level",
     "years_experience": None,
     "skills": [("Excel", OTHER_CLUSTER), ("Communication", OTHER_CLUSTER),
                ("Scheduling", OTHER_CLUSTER)]},
]

DEMO_ALIASES: dict[str, str] = {
    "js": "JavaScript",
    "k8s": "Kubernetes",
    "reactjs": "React",
    "vuejs": "Vue",
    "postgres": "SQL",
}

_FORM_ONLY_SKILLS: list[tuple[str, str]] = [("HTML", "Front-end"), ("TypeScript", "Front-end")]


def _row_skills() -> list[tuple[str, str]]:
    rows = DEMO_RESUMES + DEMO_POSTINGS
    return [pair for row in rows for pair in row["skills"]] + _FORM_ONLY_SKILLS


DEMO_CLUSTERS: dict[str, str] = {normalize.normalize_key(s): c for s, c in _row_skills()}
_SPELLINGS: dict[str, str] = {normalize.normalize_key(s): s for s, _ in _row_skills()}


def _view_from_row(row: dict, side: models.Side) -> models.DocumentView:
    is_resume = side == "resume"
    return models.DocumentView(
        id=row["id"],
        side=side,
        title=row["title"],
        level=row["level"],
        level_source="years" if is_resume else "listed",
        years_experience=row["years_experience"],
        skills=[_doc_skill(skill, cluster) for skill, cluster in row["skills"]],
    )


def _doc_skill(skill: str, cluster: str) -> models.DocSkill:
    via = "noise" if cluster == OTHER_CLUSTER else "taxonomy"
    return models.DocSkill(skill=skill, cluster=cluster, via=via)


def _keys(skills: list[models.DocSkill]) -> set[str]:
    return {normalize.normalize_key(s.skill) for s in skills}


def score_pair(
    system: models.SystemId, resume: list[models.DocSkill], posting: list[models.DocSkill]
) -> float:
    """Demo-only scores from skill overlap; see the I-3 formulas."""
    resume_keys, posting_keys = _keys(resume), _keys(posting)
    shared = len(resume_keys & posting_keys)
    if system == "B2":
        return round(shared / max(1, len(posting_keys)), 4)
    if system == "B1+":
        return round(shared / max(1, len(resume_keys | posting_keys)), 4)
    resume_clusters = {
        s.cluster for s in resume if normalize.normalize_key(s.skill) not in posting_keys
    }
    related = sum(
        1
        for key, skill in {normalize.normalize_key(s.skill): s for s in posting}.items()
        if key not in resume_keys
        and skill.cluster != OTHER_CLUSTER
        and skill.cluster in resume_clusters
    )
    return round((shared + 0.5 * related) / max(1, len(posting_keys)), 4)


def _incomplete_reason(skills: Collection[models.DocSkill]) -> str | None:
    if not skills:
        return "no_skills"
    return "few_skills" if len(skills) < FEW_SKILLS else None


def _notes(query: models.DocumentView) -> list[models.Note]:
    notes: list[models.Note] = []
    if not query.skills:
        notes.append(models.Note(code="no_skills", message=NO_SKILLS_MESSAGE))
    elif query.incomplete_reason:
        notes.append(models.Note(code="sparse_query", message=SPARSE_MESSAGE))
    if query.skills and all(s.cluster == OTHER_CLUSTER for s in query.skills):
        notes.append(models.Note(code="unknown_skills", message=UNKNOWN_MESSAGE))
    return notes


def _form_level(form: models.FormQuery, side: models.Side) -> tuple[str, str]:
    """Postings use the listed level, resumes derive it from years (CC-16)."""
    if side == "posting":
        listed = levels.normalize_listed_level(form.level)
        return (listed, "listed") if listed else (levels.NOT_SPECIFIED, "missing")
    derived = levels.level_from_years(form.years_experience)
    return (derived, "years") if derived else (levels.NOT_SPECIFIED, "missing")


def _canonical_form_skills(raw: list[str]) -> list[models.DocSkill]:
    skills: list[models.DocSkill] = []
    seen: set[str] = set()
    for text in raw:
        key = normalize.normalize_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        name = DEMO_ALIASES.get(key, normalize.clean_surface(text))
        name_key = normalize.normalize_key(name)
        name = _SPELLINGS.get(name_key, name)
        cluster = DEMO_CLUSTERS.get(name_key)
        if cluster is None:
            skills.append(models.DocSkill(skill=name, cluster=OTHER_CLUSTER, via="default"))
        else:
            skills.append(_doc_skill(name, cluster))
    return skills


class DemoEngine:
    def __init__(self) -> None:
        self._views = {
            "resume": [_view_from_row(r, "resume") for r in DEMO_RESUMES],
            "posting": [_view_from_row(r, "posting") for r in DEMO_POSTINGS],
        }
        self._by_id = {v.id: v for views in self._views.values() for v in views}
        self._provenance = models.Provenance(
            alias_table_version=DEMO_VERSION,
            taxonomy_version=DEMO_VERSION,
            serialization_version=DEMO_VERSION,
            relabel_mode=DEMO_VERSION,
            embedding=None,
            demo=True,
        )

    def health(self) -> models.EngineHealth:
        return models.EngineHealth(
            mode="demo",
            artifacts_dir=None,
            counts={
                "resumes": len(self._views["resume"]),
                "postings": len(self._views["posting"]),
            },
            systems=list(SYSTEMS),
            levels=[*levels.LEVELS, levels.NOT_SPECIFIED],
            versions=self._provenance,
            capabilities={
                "structure": models.Capability(
                    available=False, reason="Demo mode has no language model."
                ),
                "embed": models.Capability(
                    available=True, detail="demo scoring, no embedding model"
                ),
            },
        )

    def examples(self, side: models.Side, limit: int) -> list[models.DocumentView]:
        return self._views[side][:limit]

    def get_document(self, doc_id: str) -> models.DocumentView:
        view = self._by_id.get(doc_id)
        if view is None:
            raise engine.EngineError("not_found", f"No document with id {doc_id!r}")
        return view

    def match(self, request: models.MatchRequest) -> engine.MatchResult:
        is_resume_query = request.direction == "resume_to_posting"
        query_side: models.Side = "resume" if is_resume_query else "posting"
        other_side: models.Side = "posting" if is_resume_query else "resume"
        query = self._query_view(request.query, query_side)
        scored = [
            (view, self._score(request.system, query_side, query, view))
            for view in self._views[other_side]
        ]
        scored.sort(key=lambda pair: -pair[1])  # stable: ties keep document position
        k_effective = min(request.k, len(scored))
        return engine.MatchResult(
            query=query,
            matches=scored[:k_effective],
            notes=_notes(query),
            provenance=self._provenance,
            k_effective=k_effective,
        )

    @staticmethod
    def _score(
        system: models.SystemId,
        query_side: models.Side,
        query: models.DocumentView,
        other: models.DocumentView,
    ) -> float:
        if query_side == "resume":
            return score_pair(system, query.skills, other.skills)
        return score_pair(system, other.skills, query.skills)

    def _query_view(self, query: models.Query, side: models.Side) -> models.DocumentView:
        if isinstance(query, models.CorpusQuery):
            return self._corpus_view(query.id, side)
        if isinstance(query, models.FormQuery):
            return self._form_view(query, side)
        raise engine.EngineError(
            "capability_unavailable",
            "Demo mode has no language model.",
            hint="Fill in the form or pick an example instead.",
        )

    def _corpus_view(self, doc_id: str, side: models.Side) -> models.DocumentView:
        view = self.get_document(doc_id)
        if view.side != side:
            raise engine.EngineError(
                "side_mismatch",
                f"Document {doc_id!r} is a {view.side}, but this search needs a {side}.",
                field="query.id",
            )
        return view

    @staticmethod
    def _form_view(form: models.FormQuery, side: models.Side) -> models.DocumentView:
        title = form.title.strip()
        skills = _canonical_form_skills(form.skills)
        if not title and not skills:
            raise engine.EngineError(
                "empty_query", "Add a title or some skills first.", field="query"
            )
        level, source = _form_level(form, side)
        return models.DocumentView(
            id=None,
            side=side,
            title=title or None,
            level=level,
            level_source=source,
            years_experience=form.years_experience if side == "resume" else None,
            skills=skills,
            incomplete_reason=_incomplete_reason(skills),
        )

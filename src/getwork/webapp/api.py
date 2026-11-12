"""FastAPI app over an Engine and an Explainer (CC-18, ADR-16, ADR-17)."""

import logging
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from getwork.llm import client as llm_client
from getwork.webapp import engine as engine_mod
from getwork.webapp import explain, models

LOGGER = logging.getLogger("getwork.webapp")
INTERNAL_MESSAGE = "Something went wrong on the server"
DEFAULT_EXAMPLES = 6
PLACEHOLDER_HTML = (
    "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
    "<title>GetWork</title></head>"
    "<body style=\"font-family: Arial, sans-serif; max-width: 40rem; margin: 2rem auto;\">"
    "<h1>GetWork</h1>"
    "<p>Frontend not built. Run <code>npm --prefix src/web run build</code>, "
    "or open <a href=\"/docs\">/docs</a> for the API.</p>"
    "</body></html>"
)


class QueryDirectionError(Exception):
    """The explain query's side disagrees with the direction (400 invalid_request)."""


def _error_response(
    status: int, code: str, message: str, *, hint: str | None = None, field: str | None = None
) -> JSONResponse:
    body = models.ErrorEnvelope(
        error=models.ErrorBody(code=code, message=message, hint=hint, field=field)
    )
    return JSONResponse(status_code=status, content=body.model_dump())


def _dotted_field(location: Sequence[object]) -> str | None:
    parts = [str(p) for p in location]
    if parts and parts[0] in ("body", "query"):
        parts = parts[1:]
    return ".".join(parts) or None


def _install_error_handlers(app: FastAPI, secrets: Sequence[str]) -> None:
    @app.exception_handler(engine_mod.EngineError)
    async def on_engine_error(_: Request, exc: engine_mod.EngineError) -> JSONResponse:
        return _error_response(exc.status, exc.code, exc.message, hint=exc.hint, field=exc.field)

    @app.exception_handler(QueryDirectionError)
    async def on_query_direction_error(_: Request, exc: QueryDirectionError) -> JSONResponse:
        return _error_response(400, "invalid_request", str(exc), field="query")

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0]
        return _error_response(
            models.ERROR_CODES["invalid_request"],
            "invalid_request",
            str(first.get("msg", "Invalid request")),
            field=None if first.get("type") == "json_invalid" else _dotted_field(first["loc"]),
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            return _error_response(404, "not_found", "Nothing exists at this address")
        response = _error_response(exc.status_code, "invalid_request", str(exc.detail))
        response.headers.update(exc.headers or {})
        return response

    @app.exception_handler(Exception)
    async def on_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        text = "".join(traceback.format_exception(exc))
        LOGGER.error("Unhandled error: %s", llm_client.redact_secrets(text, secrets))
        return _error_response(models.ERROR_CODES["internal"], "internal", INTERNAL_MESSAGE)


def _install_api_routes(
    app: FastAPI, engine: engine_mod.Engine, explainer: explain.Explainer, chat: models.ChatInfo
) -> None:
    @app.get("/api/health", response_model=models.HealthResponse)
    def health() -> models.HealthResponse:
        return models.HealthResponse(**engine.health().model_dump(), explainer=explainer.info, chat=chat)

    @app.get("/api/examples", response_model=models.ExamplesResponse)
    def examples(
        side: Literal["resume", "posting"],
        limit: int = Query(default=DEFAULT_EXAMPLES, ge=1, le=models.MAX_EXAMPLES),
    ) -> models.ExamplesResponse:
        return models.ExamplesResponse(items=engine.examples(side, limit))

    @app.post("/api/match", response_model=models.MatchResponse)
    def match(request: models.MatchRequest) -> models.MatchResponse:
        result = engine.match(request)
        matches = [
            models.Match(
                rank=position,
                id=document.id or "",
                score=score,
                document=document,
                evidence=explain.build_evidence(request.direction, result.query, document),
            )
            for position, (document, score) in enumerate(result.matches, start=1)
        ]
        return models.MatchResponse(
            direction=request.direction,
            system=request.system,
            query_kind=request.query.kind,
            k=request.k,
            k_effective=result.k_effective,
            query=result.query,
            matches=matches,
            notes=result.notes,
            provenance=result.provenance,
        )

    @app.post("/api/explain", response_model=models.Explanation)
    def explain_match(request: models.ExplainRequest) -> models.Explanation:
        expected_side = "resume" if request.direction == "resume_to_posting" else "posting"
        if request.query.side != expected_side:
            raise QueryDirectionError(
                f"Direction {request.direction} needs a {expected_side} as the query."
            )
        document = engine.get_document(request.match_id)
        if document.side == request.query.side:
            raise engine_mod.EngineError(
                "side_mismatch",
                f"Document {request.match_id!r} is on the same side as the query.",
                field="match_id",
            )
        return explainer.explain(request.direction, request.query, document)


def _install_static(app: FastAPI, static_dir: Path | None) -> None:
    if static_dir is None or not (static_dir / "index.html").is_file():

        @app.get("/", include_in_schema=False, response_class=HTMLResponse)
        def placeholder() -> HTMLResponse:
            return HTMLResponse(PLACEHOLDER_HTML)

        return

    @app.get("/", include_in_schema=False)
    def built_index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    if (static_dir / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")


def create_app(
    engine: engine_mod.Engine,
    explainer: explain.Explainer,
    *,
    static_dir: Path | None = None,
    secrets: Sequence[str] = (),
    chat: models.ChatInfo | None = None,
) -> FastAPI:
    """chat is reported in health only; the chat routes are registered separately. No CORS middleware is installed. static_dir None: GET / is the "Frontend not built" page."""
    app = FastAPI(title="GetWork")
    _install_error_handlers(app, secrets)
    _install_api_routes(app, engine, explainer, chat or models.ChatInfo())
    _install_static(app, static_dir)
    return app

"""POST /api/chat and the AgentFailure handler, added on top of the web-app app (CC-19)."""

import logging
from collections.abc import Sequence

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from getwork.agent.explainer import ChatService
from getwork.agent.failure import AgentFailure
from getwork.agent.models import AGENT_ERROR_CODE, AGENT_ERROR_STATUS, ChatRequest, ChatResponse
from getwork.llm.client import redact_secrets
from getwork.webapp import models

LOGGER = logging.getLogger("getwork.agent")


def register_chat_routes(app: FastAPI, service: ChatService, *, secrets: Sequence[str] = ()) -> None:
    """Adds POST /api/chat (response_model ChatResponse) ahead of the web-app /api/{rest:path} catch-all, and an
    exception handler for AgentFailure on the app (so /api/explain gets it too): it logs "agent failed:
    reason=<reason>" at WARNING (redact_secrets(secrets), no message text) and returns status AGENT_ERROR_STATUS
    with ErrorEnvelope(error=ErrorBody(code=AGENT_ERROR_CODE, message=failure.message, hint=failure.hint)).
    EngineError and request validation use the app's existing handlers. Any other exception becomes the existing
    generic 500 internal envelope (no message text logged)."""

    def chat(request: ChatRequest) -> ChatResponse:
        return service.chat(request)

    # The web-app 404 is an HTTP-exception handler, not a catch-all route, so appending the route is enough.
    app.add_api_route("/api/chat", chat, methods=["POST"], response_model=ChatResponse)

    async def on_agent_failure(_: Request, failure: Exception) -> JSONResponse:
        if not isinstance(failure, AgentFailure):
            raise failure
        LOGGER.warning("%s", redact_secrets(f"agent failed: reason={failure.reason}", secrets))
        body = models.ErrorEnvelope(
            error=models.ErrorBody(code=AGENT_ERROR_CODE, message=failure.message, hint=failure.hint)
        )
        return JSONResponse(status_code=AGENT_ERROR_STATUS, content=body.model_dump())

    app.add_exception_handler(AgentFailure, on_agent_failure)

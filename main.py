"""FINOVA Role-Based Banking Policy RAG API.

Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import __version__
from src.config import ROLE_DESCRIPTIONS, ConfigurationError, Role, get_settings
from src.rag_pipeline import (
    EmbeddingError,
    GenerationError,
    IndexCorruptedError,
    IndexIncompatibleError,
    IndexNotFoundError,
    RAGPipeline,
    index_status,
    is_index_ready,
    read_ingest_report,
    tenant_visible_chunk_count,
)
from src.role_manager import AuthError, AuthenticatedUser, get_current_user, require_roles

# --- Logging with request ids ------------------------------------------------------------

_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")
REQUEST_ID_HEADER = "X-Request-ID"


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


logger = logging.getLogger("finova")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s [%(request_id)s]: %(message)s")
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_RequestIdFilter())


# --- Schemas -------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Only the question is accepted. Role/tenant fields are rejected (extra='forbid')
    so a client cannot even attempt to supply its own role."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {"question": "What is the account closure process?"}})

    question: str = Field(..., min_length=3, description="Natural-language policy question.")

    @field_validator("question")
    @classmethod
    def _check_length(cls, value: str) -> str:
        value = value.strip()
        limit = get_settings().max_question_chars
        if len(value) < 3:
            raise ValueError("question must contain at least 3 non-blank characters")
        if len(value) > limit:
            raise ValueError(f"question must be at most {limit} characters")
        return value


class SourceDocument(BaseModel):
    document: str = Field(..., description="Source PDF filename.")
    pages: list[int] = Field(..., description="1-based page numbers used from this document.")


class ChatResponse(BaseModel):
    answer: str
    role: Role = Field(..., description="Backend-verified role used for retrieval.")
    found: bool = Field(..., description="False when the answer is not in the permitted documents.")
    sources: list[SourceDocument] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    source_pages: list[int] = Field(default_factory=list, description="All cited page numbers (see `sources` for per-document pages).")
    context_warning: bool = Field(False, description="True if retrieved text resembled prompt injection.")


class RoleInfo(BaseModel):
    role: Role
    description: str


class RolesResponse(BaseModel):
    roles: list[RoleInfo]


class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' when every role index is built and usable, otherwise 'degraded'.")
    version: str
    indexes_built: int
    indexes_total: int


class IngestStatusRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roles: list[Role] | None = Field(None, description="Roles to report; default all.")


class IndexStatus(BaseModel):
    role: Role
    directory_exists: bool
    index_built: bool
    embedding_model_matches: bool
    built_at: str | None = None
    embedding_model: str | None = None
    chunk_count: int | None = Field(
        None,
        description="SUPER_ADMIN: all chunks. ADMIN: only chunks visible to their tenant (GLOBAL + own tenant).",
    )
    documents: list[str] = Field(default_factory=list, description="SUPER_ADMIN only.")
    tenants: list[str] = Field(default_factory=list, description="SUPER_ADMIN only.")


class IngestRoleOutcome(BaseModel):
    status: str
    detail: str = ""


class IngestReport(BaseModel):
    finished_at: str | None = None
    configuration_error: str | None = None
    roles: dict[str, IngestRoleOutcome] = Field(default_factory=dict)


class IngestStatusResponse(BaseModel):
    indexes: list[IndexStatus]
    last_ingest: IngestReport | None = Field(
        None, description="Outcome of the last `python ingest.py` run (SUPER_ADMIN only)."
    )
    note: str = "Indexes are built offline with `python ingest.py`; this endpoint is read-only."


class ErrorResponse(BaseModel):
    detail: str


# --- App -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()  # raises on invalid configuration -> server refuses to start
    if settings.demo_auth_enabled:
        logger.warning("DEMO AUTHENTICATION IS ENABLED. Never use this outside local development.")
    # Startup diagnostics: presence flags only, never values.
    logger.info(
        "Config: app_env=%s auth_mode=%s llm_provider=%s llm_fallback=%s gemini_key_set=%s groq_key_set=%s supabase_configured=%s",
        settings.app_env.value,
        settings.auth_mode.value,
        settings.llm_provider.value,
        settings.llm_fallback_provider.value if settings.llm_fallback_provider else "none",
        settings.gemini_api_key is not None,
        settings.groq_api_key is not None,
        bool(settings.supabase_url and settings.supabase_anon_key and settings.supabase_service_role_key),
    )
    for role in Role:
        status = index_status(role, settings)
        if is_index_ready(status):
            logger.info("Index %s ready (%s chunks)", role.value, status["chunk_count"])
        else:
            logger.error(
                "Index %s NOT ready (built=%s, embedding_model_matches=%s). /chat for this role returns 503. "
                "See /ingest/status (SUPER_ADMIN) or the ingest logs above.",
                role.value, status["index_built"], status["embedding_model_matches"],
            )
    app.state.pipeline = RAGPipeline(settings)
    yield
    logger.info("Shutting down")


settings_at_import = get_settings()

app = FastAPI(
    title="FINOVA Role-Based Banking Policy RAG API",
    version=__version__,
    description=(
        "Answers banking-policy questions from documents permitted for the caller's "
        "backend-verified role. Authenticate with a Supabase access token "
        "(`Authorization: Bearer <token>`). Sample documents are fictional."
    ),
    lifespan=lifespan,
)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Attach a request id to logs and the response (reuses a sane incoming id)."""
    incoming = request.headers.get(REQUEST_ID_HEADER) or request.headers.get("X-Railway-Request-Id") or ""
    request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
    token = _request_id.set(request_id)
    try:
        try:
            response = await call_next(request)
        except Exception:
            # Handled here (not only in the Exception handler) so the log line and
            # the 500 response both carry the request id.
            logger.exception("Unhandled error")
            response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
    finally:
        _request_id.reset(token)


# Added after the request-id middleware so CORS is the outer layer and also
# decorates the 500 responses produced above.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings_at_import.cors_origin_list,
    allow_origin_regex=settings_at_import.cors_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", REQUEST_ID_HEADER],
    expose_headers=[REQUEST_ID_HEADER],
)


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


# --- Error handling --------------------------------------------------------------------

@app.exception_handler(AuthError)
async def _auth_error(_: Request, exc: AuthError) -> JSONResponse:
    headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=headers)


@app.exception_handler(IndexNotFoundError)
@app.exception_handler(IndexIncompatibleError)
@app.exception_handler(IndexCorruptedError)
async def _index_error(_: Request, exc: Exception) -> JSONResponse:
    logger.error("Index error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "The policy index for your role is not available. Please contact an administrator."})


@app.exception_handler(ConfigurationError)
async def _config_error(_: Request, exc: ConfigurationError) -> JSONResponse:
    logger.error("Configuration error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "The service is not fully configured."})


@app.exception_handler(GenerationError)
async def _generation_error(_: Request, exc: GenerationError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "The language model is temporarily unavailable."})


@app.exception_handler(EmbeddingError)
async def _embedding_error(_: Request, exc: EmbeddingError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "The embedding service is temporarily unavailable."})


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


_ERRORS = {
    401: {"model": ErrorResponse, "description": "Missing, invalid or expired token"},
    403: {"model": ErrorResponse, "description": "Inactive user or role not permitted"},
    503: {"model": ErrorResponse, "description": "Auth service, index or configuration unavailable"},
}


# --- Endpoints -----------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health(pipeline: RAGPipeline = Depends(get_pipeline)) -> HealthResponse:
    """Liveness check. Does not call external services or reveal configuration.

    Always HTTP 200 while the process is up (so Railway keeps it running and
    /ingest/status stays reachable for diagnosis); ``status`` is ``degraded``
    when any role index is missing or built with a different embedding model.
    """
    statuses = [index_status(role, pipeline.settings) for role in Role]
    built = sum(1 for s in statuses if s["index_built"])
    ready = all(is_index_ready(s) for s in statuses)
    return HealthResponse(
        status="ok" if ready else "degraded", version=__version__, indexes_built=built, indexes_total=len(Role)
    )


@app.get("/roles", response_model=RolesResponse, tags=["system"])
async def roles() -> RolesResponse:
    """List the roles recognised by the system (roles are assigned in Supabase, not here)."""
    return RolesResponse(roles=[RoleInfo(role=r, description=ROLE_DESCRIPTIONS[r]) for r in Role])


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["rag"],
    responses={**_ERRORS, 502: {"model": ErrorResponse, "description": "LLM or embedding provider failure"}},
)
async def chat(
    body: ChatRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> ChatResponse:
    """Answer a question using only documents permitted for the caller's verified role and tenant."""
    result = await pipeline.answer(body.question, user)

    grouped: dict[str, list[int]] = {}
    for ref in result.sources:
        pages = grouped.setdefault(ref.document, [])
        if ref.page not in pages:
            pages.append(ref.page)
    return ChatResponse(
        answer=result.answer,
        role=result.role,
        found=result.found,
        sources=[SourceDocument(document=d, pages=sorted(p)) for d, p in grouped.items()],
        source_documents=list(grouped),
        source_pages=sorted({ref.page for ref in result.sources}),
        context_warning=result.injection_flagged,
    )


@app.post("/ingest/status", response_model=IngestStatusResponse, tags=["admin"], responses=_ERRORS)
async def ingest_status(
    body: IngestStatusRequest | None = None,
    user: AuthenticatedUser = Depends(require_roles([Role.SUPER_ADMIN, Role.ADMIN])),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> IngestStatusResponse:
    """Report the state of each role index. Restricted to SUPER_ADMIN and ADMIN.

    Read-only: never builds, modifies or uploads anything.

    * SUPER_ADMIN: full metadata (filenames, tenant ids, total chunk counts) and
      the outcome of the last ingestion run.
    * ADMIN (tenant-level): build state plus chunk counts limited to GLOBAL and
      their own tenant; no filenames, tenant ids or other tenants' counts.
    """
    is_super = user.role == Role.SUPER_ADMIN
    selected = list(dict.fromkeys(body.roles)) if body and body.roles else list(Role)
    statuses = []
    for role in selected:
        status: dict[str, Any] = index_status(role, pipeline.settings)
        if not is_super:
            status["documents"] = []
            status["tenants"] = []
            status["chunk_count"] = tenant_visible_chunk_count(status, user.tenant_id) if status["index_built"] else None
        status.pop("tenant_chunk_counts", None)
        statuses.append(IndexStatus(**status))
    report = read_ingest_report(pipeline.settings) if is_super else None
    return IngestStatusResponse(indexes=statuses, last_ingest=IngestReport(**report) if report else None)

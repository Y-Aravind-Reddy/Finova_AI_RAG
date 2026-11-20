"""FINOVA Role-Based Banking Policy RAG API.

Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import __version__
from src.config import ROLE_DESCRIPTIONS, ConfigurationError, Role, get_settings
from src.rag_pipeline import (
    GenerationError,
    IndexIncompatibleError,
    IndexNotFoundError,
    RAGPipeline,
    index_status,
)
from src.role_manager import AuthError, AuthenticatedUser, get_current_user, require_roles

logger = logging.getLogger("finova")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


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
    status: str
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
    chunk_count: int | None = None
    documents: list[str] = Field(default_factory=list)
    tenants: list[str] = Field(default_factory=list)


class IngestStatusResponse(BaseModel):
    indexes: list[IndexStatus]
    note: str = "Indexes are built offline with `python ingest.py`; this endpoint is read-only."


class ErrorResponse(BaseModel):
    detail: str


# --- App -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()  # raises on invalid configuration -> server refuses to start
    if settings.demo_auth_enabled:
        logger.warning("DEMO AUTHENTICATION IS ENABLED. Never use this outside local development.")
    app.state.pipeline = RAGPipeline(settings)
    yield


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings_at_import.cors_origin_list,
    allow_origin_regex=settings_at_import.cors_origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
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
    """Liveness check. Does not call external services or reveal configuration."""
    built = sum(1 for role in Role if index_status(role, pipeline.settings)["index_built"])
    return HealthResponse(status="ok", version=__version__, indexes_built=built, indexes_total=len(Role))


@app.get("/roles", response_model=RolesResponse, tags=["system"])
async def roles() -> RolesResponse:
    """List the roles recognised by the system (roles are assigned in Supabase, not here)."""
    return RolesResponse(roles=[RoleInfo(role=r, description=ROLE_DESCRIPTIONS[r]) for r in Role])


@app.post("/chat", response_model=ChatResponse, tags=["rag"], responses={**_ERRORS, 502: {"model": ErrorResponse}})
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

    Document filenames and tenant ids span all tenants, so they are only shown
    to SUPER_ADMIN; a (tenant-level) ADMIN sees counts and build state only.
    """
    selected = (body.roles if body and body.roles else None) or list(Role)
    statuses = []
    for role in selected:
        status = index_status(role, pipeline.settings)
        if user.role != Role.SUPER_ADMIN:
            status["documents"] = []
            status["tenants"] = []
        statuses.append(IndexStatus(**status))
    return IngestStatusResponse(indexes=statuses)

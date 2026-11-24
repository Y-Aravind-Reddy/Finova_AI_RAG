"""Role-specific retrieval and grounded response generation.

Chunks and embeddings live in Supabase (``src/vector_store.py``). Access
enforcement happens in two independent layers:

1. **In the database** - ``match_policy_chunks`` only returns rows whose
   ``role_scope``/``allowed_roles`` match the user's backend-verified role and
   whose ``tenant_id`` is ``GLOBAL`` or the user's own tenant.
2. **In the API** - every returned chunk is checked again with
   ``is_document_permitted`` (defence in depth). A chunk without valid
   metadata is dropped.

LIMITATION: all tenants' chunks for a role share one table; isolation is by
``tenant_id`` in the SQL filter, not separate storage. See README "Security
limitations".
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from src.chains import build_rag_chain, get_llm_with_fallback
from src.config import GLOBAL_TENANT_ID, Role, Settings, get_settings
from src.embeddings import embedding_model_id, get_embeddings
from src.prompts import NOT_FOUND_MESSAGE, NOT_FOUND_SENTINEL, detect_injection, prompt_variables
from src.role_manager import AuthenticatedUser
from src.vector_store import SupabasePolicyStore

logger = logging.getLogger(__name__)

# Matches inline citations such as "[policy.pdf, p. 2]" or "[source=policy.pdf, p. 2]".
_CITATION_RE = re.compile(r"([\w.\-]+\.pdf)\s*,?\s*p(?:age)?\.?\s*(\d+)", re.IGNORECASE)
# How long /chat and /health may reuse the policy_indexes rows (one small query).
INDEX_CACHE_SECONDS = 30.0


class RAGError(RuntimeError):
    """Base class for pipeline errors."""


class IndexNotFoundError(RAGError):
    """The role's index has not been built yet."""


class IndexIncompatibleError(RAGError):
    """The index was built with a different embedding model than is configured."""


class EmbeddingError(RAGError):
    """The embedding provider failed to embed the question."""


class GenerationError(RAGError):
    """The LLM call failed."""


@dataclass(frozen=True)
class SourceRef:
    document: str
    page: int


@dataclass
class RAGResult:
    answer: str
    role: Role
    found: bool
    sources: list[SourceRef] = field(default_factory=list)
    injection_flagged: bool = False
    provider: str | None = None


# --- Index state (rows of policy_indexes) -------------------------------------------------

def index_status(role: Role, row: dict[str, Any] | None, settings: Settings | None = None) -> dict[str, Any]:
    """Describe a role's index from its ``policy_indexes`` row (None = no row)."""
    settings = settings or get_settings()
    built = bool(row and row.get("active_build_id"))
    counts = row.get("tenant_chunk_counts") if built and row else None
    counts = counts if isinstance(counts, dict) else None
    model = row.get("embedding_model") if row else None
    return {
        "role": role.value,
        "directory_exists": row is not None,  # kept for API compatibility: "role has a row"
        "index_built": built,
        "embedding_model_matches": bool(built and model == embedding_model_id(settings)),
        "built_at": row.get("built_at") if built and row else None,
        "embedding_model": model,
        "chunk_count": row.get("chunk_count") if built and row else None,
        "documents": list(row.get("documents") or []) if built and row else [],
        "tenants": sorted(counts) if counts else [],
        "tenant_chunk_counts": counts,
    }


def is_index_ready(status: dict[str, Any]) -> bool:
    return bool(status["index_built"] and status["embedding_model_matches"])


def tenant_visible_chunk_count(status: dict[str, Any], tenant_id: str | None) -> int | None:
    """Chunks a user of ``tenant_id`` could retrieve (GLOBAL + own tenant).

    Returns None when per-tenant counts are unavailable, rather than falling
    back to the all-tenant total.
    """
    counts = status.get("tenant_chunk_counts")
    if not isinstance(counts, dict):
        return None
    total = int(counts.get(GLOBAL_TENANT_ID, 0))
    if tenant_id and tenant_id != GLOBAL_TENANT_ID:
        total += int(counts.get(tenant_id, 0))
    return total


def last_ingest_report(rows: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Outcome of the most recent ingestion run per role, from ``last_run_*`` columns."""
    runs = {
        role: {"status": row["last_run_status"], "detail": row.get("last_run_detail") or ""}
        for role, row in rows.items()
        if row.get("last_run_status")
    }
    if not runs:
        return None
    finished = [str(row["last_run_at"]) for row in rows.values() if row.get("last_run_at")]
    return {"finished_at": max(finished) if finished else None, "configuration_error": None, "roles": runs}


# --- Permission checks ------------------------------------------------------------

def is_document_permitted(metadata: dict[str, Any], user: AuthenticatedUser) -> bool:
    """True only if the chunk is explicitly permitted for the user's role and tenant."""
    role_scope = metadata.get("role_scope")
    allowed_roles = metadata.get("allowed_roles")
    tenant_id = metadata.get("tenant_id")
    if not isinstance(allowed_roles, list) or not isinstance(tenant_id, str):
        return False  # missing metadata => deny
    if role_scope not in {r.value for r in user.accessible_roles}:
        return False
    if user.role.value not in allowed_roles:
        return False
    if tenant_id == GLOBAL_TENANT_ID:
        return True
    return user.tenant_id is not None and tenant_id == user.tenant_id


# --- Pipeline -----------------------------------------------------------------------

def _chunk_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": row.get("source"),
        "source_path": row.get("source_path"),
        "page": row.get("page"),
        "role_scope": row.get("role_scope"),
        "allowed_roles": row.get("allowed_roles"),
        "tenant_id": row.get("tenant_id"),
        "doc_id": row.get("doc_id"),
        "chunk_id": row.get("chunk_id"),
    }


class RAGPipeline:
    """Answers questions for authenticated users from the Supabase policy store."""

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
        llm: Runnable | None = None,
        store: SupabasePolicyStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embeddings = embeddings
        self._llm = llm
        self.store = store or SupabasePolicyStore(self.settings)
        self._lock = threading.Lock()
        self._index_cache: tuple[float, dict[str, dict[str, Any]]] | None = None

    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = get_embeddings(self.settings)
        return self._embeddings

    @property
    def llm(self) -> Runnable:
        if self._llm is None:
            self._llm = get_llm_with_fallback(self.settings)
        return self._llm

    def index_rows(self, max_age: float = INDEX_CACHE_SECONDS) -> dict[str, dict[str, Any]]:
        """``policy_indexes`` rows, reused for up to ``max_age`` seconds."""
        now = time.monotonic()
        with self._lock:
            cached = self._index_cache
        if cached and max_age > 0 and now - cached[0] < max_age:
            return cached[1]
        rows = self.store.fetch_indexes()
        with self._lock:
            self._index_cache = (now, rows)
        return rows

    def index_statuses(
        self, roles: Iterable[Role] = tuple(Role), max_age: float = INDEX_CACHE_SECONDS
    ) -> list[dict[str, Any]]:
        rows = self.index_rows(max_age)
        return [index_status(role, rows.get(role.value), self.settings) for role in roles]

    def _require_ready(self, role: Role, rows: dict[str, dict[str, Any]]) -> None:
        status = index_status(role, rows.get(role.value), self.settings)
        if not status["index_built"]:
            raise IndexNotFoundError(f"The {role.value} index has not been built. Run `python ingest.py` first.")
        if not status["embedding_model_matches"]:
            raise IndexIncompatibleError(
                f"The {role.value} index was built with {status['embedding_model']} but "
                f"{embedding_model_id(self.settings)} is configured. Rebuild with `python ingest.py --rebuild`."
            )

    def embed_question(self, question: str) -> list[float]:
        embeddings = self.embeddings  # ConfigurationError propagates unchanged
        try:
            vector = embeddings.embed_query(question)
        except Exception as exc:  # provider SDKs raise many exception types
            logger.error("Question embedding failed: %s", type(exc).__name__)
            raise EmbeddingError("The embedding provider request failed") from exc
        if len(vector) != self.settings.embedding_dimensions:
            raise IndexIncompatibleError(
                f"Embedding has {len(vector)} dimensions but EMBEDDING_DIMENSIONS="
                f"{self.settings.embedding_dimensions} (must match the Supabase vector column)."
            )
        return vector

    def retrieve(self, question: str, user: AuthenticatedUser) -> list[tuple[Document, float]]:
        """Top-k permitted chunks for ``user`` with relevance scores, best first."""
        k = self.settings.retrieval_top_k
        # Check every permitted index first so a missing index fails before any paid API call.
        rows = self.index_rows()
        for role in user.accessible_roles:
            self._require_ready(role, rows)
        vector = self.embed_question(question)
        model = embedding_model_id(self.settings)
        results: list[tuple[Document, float]] = []
        for role in user.accessible_roles:
            for row in self.store.match(role, user.tenant_id, model, vector, k):
                doc = Document(page_content=str(row.get("content") or ""), metadata=_chunk_metadata(row))
                # Second, independent permission check.
                if not is_document_permitted(doc.metadata, user):
                    logger.warning("Dropped non-permitted chunk %s", doc.metadata.get("chunk_id"))
                    continue
                score = max(0.0, min(1.0, float(row.get("similarity") or 0.0)))
                if score >= self.settings.retrieval_score_threshold:
                    results.append((doc, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:k]

    async def answer(self, question: str, user: AuthenticatedUser) -> RAGResult:
        """Retrieve permitted context and generate a grounded, cited answer."""
        question = question.strip()
        hits = await asyncio.to_thread(self.retrieve, question, user)
        if not hits:
            return RAGResult(answer=NOT_FOUND_MESSAGE, role=user.role, found=False)

        documents = [doc for doc, _ in hits]
        flagged = any(detect_injection(doc.page_content) for doc in documents)
        if flagged:
            logger.warning("Possible prompt injection in retrieved context for role %s", user.role.value)

        chain = build_rag_chain(self.llm)
        try:
            raw = await chain.ainvoke(prompt_variables(user.role, question, documents))
        except Exception as exc:  # provider SDKs raise many exception types
            logger.exception("LLM generation failed")
            raise GenerationError("The language model request failed") from exc

        answer = (raw or "").strip()
        if not answer or NOT_FOUND_SENTINEL in answer:
            return RAGResult(
                answer=NOT_FOUND_MESSAGE,
                role=user.role,
                found=False,
                injection_flagged=flagged,
                provider=self.settings.llm_provider.value,
            )

        # Citations come from retrieval metadata, never from model output.
        retrieved: list[SourceRef] = []
        for doc in documents:
            ref = SourceRef(document=str(doc.metadata.get("source")), page=int(doc.metadata.get("page", 0)))
            if ref not in retrieved:
                retrieved.append(ref)
        # Narrow to the retrieved chunks the model actually cited. Only refs that
        # exist in `retrieved` survive, so the model cannot add sources.
        cited = {
            (name, int(page)) for name, page in _CITATION_RE.findall(answer)
        }
        sources = [r for r in retrieved if (r.document, r.page) in cited] or retrieved
        return RAGResult(
            answer=answer,
            role=user.role,
            found=True,
            sources=sources,
            injection_flagged=flagged,
            provider=self.settings.llm_provider.value,
        )

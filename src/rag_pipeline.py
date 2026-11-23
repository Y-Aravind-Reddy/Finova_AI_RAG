"""Role-specific retrieval and grounded response generation.

Access enforcement happens in two independent layers:

1. **Index selection** – a user can only open the FAISS index(es) listed for
   their backend-verified role in ``ROLE_INDEX_ACCESS``.
2. **Per-chunk permission check** – every retrieved chunk is checked against
   ``allowed_roles`` and ``tenant_id`` metadata both inside the FAISS filter and
   again after retrieval (defence in depth). A chunk without valid metadata is
   dropped.

LIMITATION: all tenants' documents for a role share one FAISS index file, and
tenant filtering is a metadata post-filter. This prevents cross-tenant results
from being returned but is not physical isolation. See README "Security
limitations".
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from src.chains import build_rag_chain, get_llm_with_fallback
from src.config import GLOBAL_TENANT_ID, Role, Settings, get_settings
from src.embeddings import embedding_model_id, get_embeddings
from src.prompts import NOT_FOUND_MESSAGE, NOT_FOUND_SENTINEL, detect_injection, prompt_variables
from src.role_manager import AuthenticatedUser

logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.json"
# Written by ``ingest.py`` next to the role folders; records why the last run failed.
INGEST_REPORT_FILE = "ingest_report.json"
# Matches inline citations such as "[policy.pdf, p. 2]" or "[source=policy.pdf, p. 2]".
_CITATION_RE = re.compile(r"([\w.\-]+\.pdf)\s*,?\s*p(?:age)?\.?\s*(\d+)", re.IGNORECASE)
INDEX_FILES = ("index.faiss", "index.pkl")


class RAGError(RuntimeError):
    """Base class for pipeline errors."""


class IndexNotFoundError(RAGError):
    """The role's index has not been built yet."""


class IndexIncompatibleError(RAGError):
    """The index was built with a different embedding model than is configured."""


class IndexCorruptedError(RAGError):
    """The index files exist but cannot be loaded."""


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


# --- Index files -----------------------------------------------------------------

def role_index_dir(role: Role, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.vectorstore_dir / role.folder_name


def read_manifest(index_dir: Path) -> dict[str, Any] | None:
    path = index_dir / MANIFEST_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Unreadable manifest %s: %s", path, exc)
        return None


def index_status(role: Role, settings: Settings | None = None) -> dict[str, Any]:
    """Describe the on-disk state of a role's index (no network calls)."""
    settings = settings or get_settings()
    index_dir = role_index_dir(role, settings)
    files_present = all((index_dir / name).is_file() for name in INDEX_FILES)
    manifest = read_manifest(index_dir)
    built = files_present and manifest is not None
    compatible = bool(built and manifest and manifest.get("embedding_model") == embedding_model_id(settings))
    return {
        "role": role.value,
        "directory_exists": index_dir.is_dir(),
        "index_built": built,
        "embedding_model_matches": compatible,
        "built_at": manifest.get("built_at") if manifest else None,
        "embedding_model": manifest.get("embedding_model") if manifest else None,
        "chunk_count": manifest.get("chunk_count") if manifest else None,
        "documents": manifest.get("documents", []) if manifest else [],
        "tenants": manifest.get("tenants", []) if manifest else [],
        "tenant_chunk_counts": manifest.get("tenant_chunk_counts") if manifest else None,
    }


def is_index_ready(status: dict[str, Any]) -> bool:
    return bool(status["index_built"] and status["embedding_model_matches"])


def tenant_visible_chunk_count(status: dict[str, Any], tenant_id: str | None) -> int | None:
    """Chunks a user of ``tenant_id`` could retrieve (GLOBAL + own tenant).

    Returns None for indexes built before per-tenant counts were recorded, rather
    than falling back to the all-tenant total.
    """
    counts = status.get("tenant_chunk_counts")
    if not isinstance(counts, dict):
        return None
    total = int(counts.get(GLOBAL_TENANT_ID, 0))
    if tenant_id and tenant_id != GLOBAL_TENANT_ID:
        total += int(counts.get(tenant_id, 0))
    return total


def read_ingest_report(settings: Settings | None = None) -> dict[str, Any] | None:
    settings = settings or get_settings()
    path = settings.vectorstore_dir / INGEST_REPORT_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Unreadable ingest report %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


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


def distance_to_relevance(distance: float) -> float:
    """Convert squared L2 distance between unit vectors to cosine similarity in [0, 1]."""
    return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))


# --- Pipeline -----------------------------------------------------------------------

class RAGPipeline:
    """Loads role indexes lazily and answers questions for authenticated users."""

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
        llm: Runnable | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._embeddings = embeddings
        self._llm = llm
        self._stores: dict[Role, tuple[str, FAISS]] = {}
        self._lock = threading.Lock()

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

    def load_store(self, role: Role) -> FAISS:
        """Load (and cache) the FAISS index for ``role``, reloading after a rebuild."""
        index_dir = role_index_dir(role, self.settings)
        manifest = read_manifest(index_dir)
        if manifest is None or not all((index_dir / f).is_file() for f in INDEX_FILES):
            raise IndexNotFoundError(
                f"The {role.value} index has not been built. Run `python ingest.py` first."
            )
        expected = embedding_model_id(self.settings)
        if manifest.get("embedding_model") != expected:
            raise IndexIncompatibleError(
                f"The {role.value} index was built with {manifest.get('embedding_model')} but "
                f"{expected} is configured. Rebuild with `python ingest.py --rebuild`."
            )
        version = str(manifest.get("built_at"))
        embeddings = self.embeddings  # may raise ConfigurationError (-> 503), not "corrupted"
        with self._lock:
            cached = self._stores.get(role)
            if cached and cached[0] == version:
                return cached[1]
            # allow_dangerous_deserialization: the docstore is a pickle. Only
            # indexes produced by ingest.py into a protected directory may be loaded.
            try:
                store = FAISS.load_local(
                    str(index_dir),
                    embeddings,
                    allow_dangerous_deserialization=True,
                    normalize_L2=True,
                    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
                )
            except Exception as exc:  # faiss/pickle raise many types on damaged files
                logger.error("Failed to load %s index: %s", role.value, type(exc).__name__)
                raise IndexCorruptedError(
                    f"The {role.value} index could not be loaded. Rebuild with `python ingest.py --rebuild`."
                ) from exc
            self._stores[role] = (version, store)
            return store

    def embed_question(self, question: str) -> list[float]:
        embeddings = self.embeddings  # ConfigurationError propagates unchanged
        try:
            return embeddings.embed_query(question)
        except Exception as exc:  # provider SDKs raise many exception types
            logger.error("Question embedding failed: %s", type(exc).__name__)
            raise EmbeddingError("The embedding provider request failed") from exc

    def retrieve(self, question: str, user: AuthenticatedUser) -> list[tuple[Document, float]]:
        """Top-k permitted chunks for ``user`` with relevance scores, best first."""
        k = self.settings.retrieval_top_k
        # Load every permitted index first so a missing index fails before any API call.
        stores = [self.load_store(role) for role in user.accessible_roles]
        vector = self.embed_question(question)
        results: list[tuple[Document, float]] = []
        for store in stores:
            hits = store.similarity_search_with_score_by_vector(
                vector,
                k=k,
                fetch_k=self.settings.retrieval_fetch_k,
                filter=lambda meta: is_document_permitted(meta, user),
            )
            for doc, distance in hits:
                # Second, independent permission check.
                if not is_document_permitted(doc.metadata, user):
                    logger.warning("Dropped non-permitted chunk %s", doc.metadata.get("chunk_id"))
                    continue
                score = distance_to_relevance(distance)
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

"""Embed the PDFs in data/<role>/ and store them in the Supabase policy store.

Usage:
    python ingest.py                     # build missing/out-of-date roles
    python ingest.py --rebuild           # force rebuild of all roles
    python ingest.py --role CUSTOMER     # only one role (repeatable)
    python ingest.py --dry-run           # load + chunk only, no embeddings / API calls

Requires the Supabase migration in supabase/migrations/ and, except for
--dry-run, GEMINI_API_KEY, SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.

Rebuilds are safe: chunks are inserted under a new build id and
``activate_policy_build`` switches the role to it in one transaction, deleting
the old build only then. A failed run leaves the previous build live and
removes its own partial rows.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from src.config import ConfigurationError, Role, Settings, get_settings
from src.document_loader import DocumentLoadError, LoadResult, load_role_documents
from src.embeddings import embedding_model_id, get_embeddings
from src.vector_store import SupabasePolicyStore, VectorStoreError, vector_literal

logger = logging.getLogger("ingest")

# Bump when stored fields change in a way that requires re-embedding.
MANIFEST_VERSION = 3

# Provider error messages are stored in Supabase; scrub anything key/token-like.
_SECRET_RE = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|gsk_[0-9A-Za-z]{20,}|eyJ[0-9A-Za-z_\-]+\.[0-9A-Za-z_\-.]+|sk-[0-9A-Za-z]{20,})")


def _redact(text: str, limit: int = 300) -> str:
    return _SECRET_RE.sub("[redacted]", text)[:limit]


@dataclass
class RoleOutcome:
    role: Role
    status: str  # built | skipped | failed | dry-run
    detail: str = ""


def source_fingerprint(result: LoadResult, settings: Settings) -> str:
    """Hash of inputs that affect the stored chunks; unchanged => skip rebuild."""
    digest = hashlib.sha256()
    digest.update(f"manifest-v{MANIFEST_VERSION}".encode())
    digest.update(f"{embedding_model_id(settings)}@{settings.embedding_dimensions}".encode())
    digest.update(f"{settings.chunk_size}:{settings.chunk_overlap}".encode())
    for doc_id, path, tenant in sorted(
        {(c.metadata["doc_id"], c.metadata["source_path"], c.metadata["tenant_id"]) for c in result.chunks}
    ):
        digest.update(f"{doc_id}|{path}|{tenant}".encode())
    return digest.hexdigest()


def build_manifest(role: Role, result: LoadResult, settings: Settings, fingerprint: str) -> dict[str, Any]:
    return {
        "role": role.value,
        "embedding_model": embedding_model_id(settings),
        "embedding_dimensions": settings.embedding_dimensions,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunk_count": len(result.chunks),
        "page_count": result.pages,
        "documents": sorted(set(result.files_loaded)),
        "tenant_chunk_counts": dict(sorted(Counter(c.metadata["tenant_id"] for c in result.chunks).items())),
        "manifest_version": MANIFEST_VERSION,
        "skipped_files": result.errors,
        "fingerprint": fingerprint,
    }


def build_role_index(
    role: Role,
    result: LoadResult,
    embeddings: Embeddings,
    settings: Settings,
    store: SupabasePolicyStore,
    fingerprint: str,
) -> str:
    """Embed ``result.chunks``, upload them as a new build and make it live. Returns the build id."""
    vectors = embeddings.embed_documents([c.page_content for c in result.chunks])
    if len(vectors) != len(result.chunks):
        raise RuntimeError(f"Embedding provider returned {len(vectors)} vectors for {len(result.chunks)} chunks")
    bad = {len(v) for v in vectors if len(v) != settings.embedding_dimensions}
    if bad:
        raise ConfigurationError(
            f"Embeddings have {sorted(bad)} dimensions but EMBEDDING_DIMENSIONS={settings.embedding_dimensions} "
            "(must match the vector(N) column in the Supabase migration)."
        )

    build_id = str(uuid.uuid4())
    rows = [
        {
            "build_id": build_id,
            "role_scope": chunk.metadata["role_scope"],
            "allowed_roles": chunk.metadata["allowed_roles"],
            "tenant_id": chunk.metadata["tenant_id"],
            "chunk_id": chunk.metadata["chunk_id"],
            "doc_id": chunk.metadata["doc_id"],
            "source": chunk.metadata["source"],
            "source_path": chunk.metadata["source_path"],
            "page": int(chunk.metadata["page"]),
            "content": chunk.page_content,
            "embedding": vector_literal(vector),
        }
        for chunk, vector in zip(result.chunks, vectors)
    ]
    try:
        store.insert_chunks(rows)
        store.activate_build(role, build_id, build_manifest(role, result, settings, fingerprint))
    except Exception:
        try:  # remove this run's partial rows; the previous build stays live
            store.delete_build(role, build_id)
        except Exception as cleanup_exc:  # noqa: BLE001
            logger.warning("Could not remove partial build %s for %s: %s", build_id, role.value, type(cleanup_exc).__name__)
        raise
    return build_id


def _record(store: SupabasePolicyStore, outcome: RoleOutcome) -> None:
    """Best-effort: store the outcome so SUPER_ADMIN can read it via /ingest/status."""
    try:
        store.record_run(outcome.role, outcome.status, _redact(outcome.detail), datetime.now(timezone.utc).isoformat())
    except (VectorStoreError, ConfigurationError) as exc:
        logger.warning("Could not record ingest outcome for %s: %s", outcome.role.value, exc)


def run_ingestion(
    settings: Settings,
    roles: list[Role],
    rebuild: bool = False,
    strict: bool = False,
    dry_run: bool = False,
    embeddings: Embeddings | None = None,
    store: SupabasePolicyStore | None = None,
) -> list[RoleOutcome]:
    """Ingest each role independently; one role's failure does not stop the others.

    Raises:
        ConfigurationError: missing Gemini key or Supabase settings.
        VectorStoreError: Supabase unreachable or the migration is not applied.
    """
    existing: dict[str, dict[str, Any]] = {}
    if not dry_run:
        settings.require_supabase_service()  # fail before any paid embedding call
        if embeddings is None:
            embeddings = get_embeddings(settings)  # raises ConfigurationError if key missing
        store = store or SupabasePolicyStore(settings)
        existing = store.fetch_indexes()

    outcomes: list[RoleOutcome] = []
    for role in roles:
        outcome = _ingest_role(role, settings, rebuild, strict, dry_run, embeddings, store, existing)
        outcomes.append(outcome)
        if store is not None and not dry_run:
            _record(store, outcome)
    return outcomes


def _ingest_role(
    role: Role,
    settings: Settings,
    rebuild: bool,
    strict: bool,
    dry_run: bool,
    embeddings: Embeddings | None,
    store: SupabasePolicyStore | None,
    existing: dict[str, dict[str, Any]],
) -> RoleOutcome:
    try:
        result = load_role_documents(role, settings.data_dir, settings.chunk_size, settings.chunk_overlap, strict=strict)
    except DocumentLoadError as exc:
        return RoleOutcome(role, "failed", str(exc))

    summary = f"{len(result.files_loaded)} PDFs, {result.pages} pages, {len(result.chunks)} chunks"
    if result.errors:
        summary += f"; skipped invalid: {', '.join(result.errors)}"
    if dry_run:
        return RoleOutcome(role, "dry-run", summary)

    fingerprint = source_fingerprint(result, settings)
    row = existing.get(role.value) or {}
    if not rebuild and row.get("active_build_id") and row.get("fingerprint") == fingerprint:
        return RoleOutcome(role, "skipped", "index up to date (use --rebuild to force)")

    try:
        assert embeddings is not None and store is not None
        build_role_index(role, result, embeddings, settings, store, fingerprint)
        return RoleOutcome(role, "built", summary)
    except Exception as exc:  # embedding API / Supabase errors; previous build stays live
        logger.exception("Failed to build index for %s", role.value)
        return RoleOutcome(role, "failed", _redact(f"{type(exc).__name__}: {exc}"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed role PDFs into the Supabase policy store.")
    parser.add_argument(
        "--role",
        action="append",
        choices=[r.value for r in Role],
        help="Role to ingest (repeatable). Default: all roles.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild even if the index is up to date.")
    parser.add_argument("--strict", action="store_true", help="Fail a role on the first invalid PDF.")
    parser.add_argument("--dry-run", action="store_true", help="Load and chunk only; no embedding calls.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # httpx logs every request URL at INFO; keep ingestion output readable.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2

    roles = [Role(r) for r in args.role] if args.role else list(Role)
    try:
        outcomes = run_ingestion(settings, roles, args.rebuild, args.strict, args.dry_run)
    except ConfigurationError as exc:
        print(f"Configuration error: {_redact(str(exc))}", file=sys.stderr)
        return 2
    except VectorStoreError as exc:
        print(f"Supabase vector store error: {_redact(str(exc))}", file=sys.stderr)
        return 2

    print("\nIngestion summary")
    print("-" * 60)
    for outcome in outcomes:
        print(f"{outcome.role.value:<18} {outcome.status:<8} {outcome.detail}")
    return 1 if any(o.status == "failed" for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())

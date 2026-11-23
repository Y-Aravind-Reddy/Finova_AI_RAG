"""Build one FAISS index per role from the PDFs in data/<role>/.

Usage:
    python ingest.py                     # build missing/out-of-date indexes for all roles
    python ingest.py --rebuild           # force rebuild of all roles
    python ingest.py --role CUSTOMER     # only one role (repeatable)
    python ingest.py --dry-run           # load + chunk only, no embeddings / API calls

Rebuilds are safe: a new index is written to a temporary directory and swapped
in only after it has been fully saved, so a failed run never destroys the
previous working index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from src.config import ConfigurationError, Role, Settings, get_settings
from src.document_loader import DocumentLoadError, LoadResult, load_role_documents
from src.embeddings import embedding_model_id, get_embeddings
from src.rag_pipeline import INGEST_REPORT_FILE, MANIFEST_FILE, read_manifest, role_index_dir

logger = logging.getLogger("ingest")

# Bump when the manifest gains fields the API relies on; forces a one-time rebuild.
MANIFEST_VERSION = 2

# Provider error messages are stored in the report; scrub anything key/token-like.
_SECRET_RE = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|gsk_[0-9A-Za-z]{20,}|eyJ[0-9A-Za-z_\-]+\.[0-9A-Za-z_\-.]+|sk-[0-9A-Za-z]{20,})")


def _redact(text: str, limit: int = 300) -> str:
    return _SECRET_RE.sub("[redacted]", text)[:limit]


@dataclass
class RoleOutcome:
    role: Role
    status: str  # built | skipped | failed | dry-run
    detail: str = ""


def source_fingerprint(result: LoadResult, settings: Settings) -> str:
    """Hash of inputs that affect the index; unchanged fingerprint => skip rebuild."""
    digest = hashlib.sha256()
    digest.update(f"manifest-v{MANIFEST_VERSION}".encode())
    digest.update(embedding_model_id(settings).encode())
    digest.update(f"{settings.chunk_size}:{settings.chunk_overlap}".encode())
    for doc_id, path, tenant in sorted(
        {(c.metadata["doc_id"], c.metadata["source_path"], c.metadata["tenant_id"]) for c in result.chunks}
    ):
        digest.update(f"{doc_id}|{path}|{tenant}".encode())
    return digest.hexdigest()


def _swap_into_place(build_dir: Path, target_dir: Path) -> None:
    """Replace ``target_dir`` with ``build_dir`` keeping a backup until success."""
    backup_dir = target_dir.with_name(f".{target_dir.name}.old-{int(time.time() * 1000)}")
    if target_dir.exists():
        target_dir.rename(backup_dir)
    try:
        build_dir.rename(target_dir)
    except OSError:
        if backup_dir.exists():
            backup_dir.rename(target_dir)  # restore previous index
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)


def build_role_index(
    role: Role,
    result: LoadResult,
    embeddings: Embeddings,
    settings: Settings,
    fingerprint: str,
) -> Path:
    """Embed ``result.chunks`` and atomically save the index for ``role``."""
    target_dir = role_index_dir(role, settings)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    build_dir = target_dir.with_name(f".{target_dir.name}.building-{int(time.time() * 1000)}")
    build_dir.mkdir(parents=True)
    try:
        store = FAISS.from_documents(
            result.chunks,
            embeddings,
            normalize_L2=True,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        store.save_local(str(build_dir))
        manifest = {
            "role": role.value,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "embedding_model": embedding_model_id(settings),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "chunk_count": len(result.chunks),
            "page_count": result.pages,
            "documents": sorted(set(result.files_loaded)),
            "tenants": sorted({c.metadata["tenant_id"] for c in result.chunks}),
            "tenant_chunk_counts": dict(sorted(Counter(c.metadata["tenant_id"] for c in result.chunks).items())),
            "manifest_version": MANIFEST_VERSION,
            "skipped_files": result.errors,
            "fingerprint": fingerprint,
        }
        (build_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (build_dir / ".gitkeep").touch()
        _swap_into_place(build_dir, target_dir)
    finally:
        if build_dir.exists():
            shutil.rmtree(build_dir, ignore_errors=True)
    return target_dir


def run_ingestion(
    settings: Settings,
    roles: list[Role],
    rebuild: bool = False,
    strict: bool = False,
    dry_run: bool = False,
    embeddings: Embeddings | None = None,
) -> list[RoleOutcome]:
    """Ingest each role independently; one role's failure does not stop the others."""
    if not dry_run and embeddings is None:
        embeddings = get_embeddings(settings)  # raises ConfigurationError if key missing

    outcomes: list[RoleOutcome] = []
    for role in roles:
        try:
            result = load_role_documents(
                role, settings.data_dir, settings.chunk_size, settings.chunk_overlap, strict=strict
            )
        except DocumentLoadError as exc:
            outcomes.append(RoleOutcome(role, "failed", str(exc)))
            continue

        summary = f"{len(result.files_loaded)} PDFs, {result.pages} pages, {len(result.chunks)} chunks"
        if result.errors:
            summary += f"; skipped invalid: {', '.join(result.errors)}"

        if dry_run:
            outcomes.append(RoleOutcome(role, "dry-run", summary))
            continue

        fingerprint = source_fingerprint(result, settings)
        existing = read_manifest(role_index_dir(role, settings))
        if not rebuild and existing and existing.get("fingerprint") == fingerprint:
            outcomes.append(RoleOutcome(role, "skipped", "index up to date (use --rebuild to force)"))
            continue

        try:
            assert embeddings is not None
            build_role_index(role, result, embeddings, settings, fingerprint)
            outcomes.append(RoleOutcome(role, "built", summary))
        except Exception as exc:  # embedding API / disk errors; keep previous index
            logger.exception("Failed to build index for %s", role.value)
            outcomes.append(RoleOutcome(role, "failed", _redact(f"{type(exc).__name__}: {exc}")))
    return outcomes


def write_ingest_report(
    settings: Settings, outcomes: list[RoleOutcome], configuration_error: str | None = None
) -> None:
    """Persist the run outcome so SUPER_ADMIN can see *why* indexes are missing
    via /ingest/status (container logs are otherwise the only record)."""
    report = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "configuration_error": _redact(configuration_error) if configuration_error else None,
        "roles": {o.role.value: {"status": o.status, "detail": _redact(o.detail)} for o in outcomes},
    }
    try:
        settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        (settings.vectorstore_dir / INGEST_REPORT_FILE).write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write ingest report: %s", exc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build role-wise FAISS indexes from PDFs.")
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
    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        return 2

    roles = [Role(r) for r in args.role] if args.role else list(Role)
    try:
        outcomes = run_ingestion(settings, roles, args.rebuild, args.strict, args.dry_run)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        if not args.dry_run:
            write_ingest_report(settings, [], configuration_error=str(exc))
        return 2
    if not args.dry_run:
        write_ingest_report(settings, outcomes)

    print("\nIngestion summary")
    print("-" * 60)
    for outcome in outcomes:
        print(f"{outcome.role.value:<18} {outcome.status:<8} {outcome.detail}")
    return 1 if any(o.status == "failed" for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())

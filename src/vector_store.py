"""Supabase (Postgres + pgvector) storage for role-scoped policy chunks.

Schema: ``supabase/migrations/20261124000000_policy_vector_store.sql``

* ``policy_chunks``  – one row per chunk: text, embedding and access metadata.
* ``policy_indexes`` – one row per role: live build id, embedding model,
  counts and the outcome of the last ingestion run.
* ``match_policy_chunks()`` – similarity search; the role, tenant, live build
  and embedding-model filters run inside the SQL query.
* ``activate_policy_build()`` – atomically switches a role to a new build and
  deletes older builds, so a failed ingestion never replaces a working one.

Only the backend calls these, with the service-role key. RLS is enabled with
no policies, so the anon key or a user JWT cannot read policy text.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import httpx

from src.config import Role, Settings

logger = logging.getLogger(__name__)

CHUNKS_TABLE = "policy_chunks"
INDEXES_TABLE = "policy_indexes"
MATCH_FUNCTION = "match_policy_chunks"
ACTIVATE_FUNCTION = "activate_policy_build"
INSERT_BATCH_SIZE = 50
INDEX_COLUMNS = (
    "role,active_build_id,built_at,embedding_model,fingerprint,chunk_count,"
    "documents,tenant_chunk_counts,last_run_at,last_run_status,last_run_detail"
)
# PostgREST/Postgres codes meaning "table or function does not exist".
_MISSING_OBJECT_CODES = {"PGRST202", "PGRST205", "42P01", "42883"}
MIGRATION_HINT = "apply supabase/migrations/20261124000000_policy_vector_store.sql"


class VectorStoreError(RuntimeError):
    """The Supabase vector store is unreachable, not migrated, or rejected a request."""


def vector_literal(vector: Sequence[float]) -> str:
    """pgvector text form, e.g. ``[0.1,0.2]`` (accepted by PostgREST for vector columns)."""
    return "[" + ",".join(format(float(x), ".9g") for x in vector) + "]"


def _error_code(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""
    return str(body.get("code", "")) if isinstance(body, dict) else ""


class SupabasePolicyStore:
    """Synchronous PostgREST client for the policy vector tables.

    Sync on purpose: retrieval runs in a worker thread (``asyncio.to_thread``)
    and ingestion is a CLI.
    """

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json: Any = None,
        prefer: str | None = None,
    ) -> httpx.Response:
        url, key = self._settings.require_supabase_service()  # ConfigurationError -> 503
        headers = {"apikey": key, "Authorization": f"Bearer {key}", "Accept": "application/json"}
        if prefer:
            headers["Prefer"] = prefer
        try:
            with httpx.Client(timeout=self._settings.supabase_timeout_seconds, transport=self._transport) as client:
                response = client.request(method, f"{url}/rest/v1/{path}", params=params, json=json, headers=headers)
        except httpx.HTTPError as exc:
            logger.error("Supabase vector store %s %s failed: %s", method, path, type(exc).__name__)
            raise VectorStoreError(f"Supabase vector store unreachable ({type(exc).__name__})") from exc
        if response.status_code >= 300:
            code = _error_code(response)
            hint = f"; {MIGRATION_HINT}" if code in _MISSING_OBJECT_CODES or response.status_code == 404 else ""
            # The body is not logged: it can echo chunk text.
            logger.error("Supabase vector store %s %s: HTTP %s %s%s", method, path, response.status_code, code, hint)
            raise VectorStoreError(f"Supabase vector store error (HTTP {response.status_code} {code}{hint})")
        return response

    # --- Reads ---------------------------------------------------------------------

    def fetch_indexes(self) -> dict[str, dict[str, Any]]:
        """All ``policy_indexes`` rows keyed by role value."""
        response = self._request("GET", INDEXES_TABLE, params={"select": INDEX_COLUMNS})
        try:
            rows = response.json()
        except ValueError as exc:
            raise VectorStoreError("Supabase vector store returned a non-JSON body") from exc
        if not isinstance(rows, list):
            raise VectorStoreError("Supabase vector store returned an unexpected body")
        return {str(row["role"]): row for row in rows if isinstance(row, dict) and row.get("role")}

    def match(
        self,
        role: Role,
        tenant_id: str | None,
        embedding_model: str,
        vector: Sequence[float],
        k: int,
    ) -> list[dict[str, Any]]:
        """Top-``k`` chunks permitted for (role, tenant), best first, with ``similarity``."""
        response = self._request(
            "POST",
            f"rpc/{MATCH_FUNCTION}",
            json={
                "p_role": role.value,
                "p_tenant_id": tenant_id,
                "p_embedding_model": embedding_model,
                "p_query_embedding": vector_literal(vector),
                "p_match_count": k,
            },
        )
        try:
            rows = response.json()
        except ValueError as exc:
            raise VectorStoreError("Supabase vector store returned a non-JSON body") from exc
        if not isinstance(rows, list):
            raise VectorStoreError("Supabase vector store returned an unexpected body")
        return [row for row in rows if isinstance(row, dict)]

    # --- Writes (ingestion only) ---------------------------------------------------------

    def insert_chunks(self, rows: Sequence[dict[str, Any]]) -> None:
        for start in range(0, len(rows), INSERT_BATCH_SIZE):
            self._request("POST", CHUNKS_TABLE, json=list(rows[start : start + INSERT_BATCH_SIZE]), prefer="return=minimal")

    def activate_build(self, role: Role, build_id: str, manifest: dict[str, Any]) -> None:
        self._request(
            "POST",
            f"rpc/{ACTIVATE_FUNCTION}",
            json={"p_role": role.value, "p_build_id": build_id, "p_manifest": manifest},
        )

    def delete_build(self, role: Role, build_id: str) -> None:
        self._request(
            "DELETE",
            CHUNKS_TABLE,
            params={"role_scope": f"eq.{role.value}", "build_id": f"eq.{build_id}"},
            prefer="return=minimal",
        )

    def record_run(self, role: Role, status: str, detail: str, finished_at: str) -> None:
        """Upsert only the last_run_* columns; the live build is left untouched."""
        self._request(
            "POST",
            INDEXES_TABLE,
            params={"on_conflict": "role"},
            json={"role": role.value, "last_run_at": finished_at, "last_run_status": status, "last_run_detail": detail},
            prefer="resolution=merge-duplicates,return=minimal",
        )

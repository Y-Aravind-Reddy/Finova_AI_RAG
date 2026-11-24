"""Shared fixtures. No test makes network calls or needs API keys."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Never let a developer's real .env/OS settings leak into tests.
os.environ["APP_ENV"] = "test"
os.environ["AUTH_MODE"] = "supabase"
os.environ["ALLOW_DEMO_AUTH"] = "false"
# OS env vars override .env, and blank values become None: tests never reach a real Supabase project.
for _name in ("SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
    os.environ[_name] = ""

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import Field  # noqa: E402

from src.config import GLOBAL_TENANT_ID, Settings  # noqa: E402
from src.vector_store import SupabasePolicyStore  # noqa: E402

SAMPLE_DATA = ROOT / "data"


class KeywordEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings (offline stand-in for Gemini)."""

    dim = 512

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"[a-z]{3,}", text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FailingEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("simulated embedding API outage")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("simulated embedding API outage")


class RecordingChatModel(BaseChatModel):
    """Returns a fixed response and records the prompts it received."""

    response: str = "Grounded answer [source, p. 1]"
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.calls.append(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])

    @property
    def _llm_type(self) -> str:
        return "recording-fake"


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"app_env": "test", "auth_mode": "supabase", "allow_demo_auth": False}
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def data_copy(tmp_path: Path) -> Path:
    """A writable copy of the sample data folder."""
    target = tmp_path / "data"
    shutil.copytree(SAMPLE_DATA, target)
    return target


# Fake Supabase project used by the vector store in tests (never a real project).
MOCK_SUPABASE = {
    "supabase_url": "https://mock-project.supabase.co",
    "supabase_anon_key": "anon",
    "supabase_service_role_key": "service",
}


def _parse_vector(value: Any) -> list[float]:
    return [float(x) for x in json.loads(value)] if isinstance(value, str) else [float(x) for x in value]


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


class FakePolicyDB:
    """In-memory stand-in for the Supabase policy tables and SQL functions.

    Served to the real ``SupabasePolicyStore`` through ``httpx.MockTransport``,
    so the production HTTP code runs in tests. The filters mirror
    supabase/migrations/20261124000000_policy_vector_store.sql exactly.
    ``fail`` can contain "down", "insert", "activate", "match" or "missing".
    """

    def __init__(self) -> None:
        self.indexes: dict[str, dict[str, Any]] = {}
        self.chunks: list[dict[str, Any]] = []
        self.fail: set[str] = set()
        self.requests: list[tuple[str, str]] = []

    @staticmethod
    def _blank_row(role: str) -> dict[str, Any]:
        return {
            "role": role, "active_build_id": None, "built_at": None, "embedding_model": None,
            "fingerprint": None, "chunk_count": 0, "documents": [], "tenant_chunk_counts": {},
            "last_run_at": None, "last_run_status": None, "last_run_detail": None,
        }

    def _match(self, body: dict[str, Any]) -> list[dict[str, Any]]:
        role, tenant = body["p_role"], body["p_tenant_id"]
        index = self.indexes.get(role) or {}
        query = _parse_vector(body["p_query_embedding"])
        rows = [
            c for c in self.chunks
            if c["role_scope"] == role
            and role in c["allowed_roles"]
            and (c["tenant_id"] == GLOBAL_TENANT_ID or (tenant is not None and c["tenant_id"] == tenant))
            and c["build_id"] == index.get("active_build_id")
            and index.get("embedding_model") == body["p_embedding_model"]
        ]
        scored = [({k: v for k, v in c.items() if k not in ("embedding", "build_id")}, _cosine(c["embedding"], query)) for c in rows]
        scored.sort(key=lambda item: item[1], reverse=True)
        limit = min(max(int(body["p_match_count"] or 5), 1), 50)
        return [{**row, "similarity": sim} for row, sim in scored[:limit]]

    def _activate(self, body: dict[str, Any]) -> httpx.Response:
        role, build, manifest = body["p_role"], body["p_build_id"], body["p_manifest"]
        count = sum(1 for c in self.chunks if c["role_scope"] == role and c["build_id"] == build)
        if count == 0 or count != manifest["chunk_count"] or "activate" in self.fail:
            return httpx.Response(400, json={"code": "P0001", "message": "build incomplete"})
        row = self.indexes.setdefault(role, self._blank_row(role))
        row.update(
            active_build_id=build, built_at="2026-11-24T00:00:00+00:00", embedding_model=manifest["embedding_model"],
            fingerprint=manifest["fingerprint"], chunk_count=count, documents=list(manifest["documents"]),
            tenant_chunk_counts=dict(manifest["tenant_chunk_counts"]),
        )
        self.chunks = [c for c in self.chunks if not (c["role_scope"] == role and c["build_id"] != build)]
        return httpx.Response(204)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        if "down" in self.fail:
            raise httpx.ConnectError("down")
        assert request.headers["apikey"] == "service" and request.headers["authorization"] == "Bearer service"
        if "missing" in self.fail:
            return httpx.Response(404, json={"code": "PGRST205", "message": "Could not find the table"})
        path = request.url.path.removeprefix("/rest/v1/")
        body = json.loads(request.content) if request.content else None
        method = request.method
        if path == "policy_indexes" and method == "GET":
            return httpx.Response(200, json=[dict(r) for r in self.indexes.values()])
        if path == "policy_indexes" and method == "POST":  # upsert of last_run_* columns
            assert request.url.params["on_conflict"] == "role"
            self.indexes.setdefault(body["role"], self._blank_row(body["role"])).update(body)
            return httpx.Response(201)
        if path == "policy_chunks" and method == "POST":
            if "insert" in self.fail:
                return httpx.Response(500, json={"code": "XX000", "message": "insert failed"})
            for row in body:
                self.chunks.append({**row, "embedding": _parse_vector(row["embedding"])})
            return httpx.Response(201)
        if path == "policy_chunks" and method == "DELETE":
            role = request.url.params["role_scope"].removeprefix("eq.")
            build = request.url.params["build_id"].removeprefix("eq.")
            self.chunks = [c for c in self.chunks if not (c["role_scope"] == role and c["build_id"] == build)]
            return httpx.Response(204)
        if path == "rpc/activate_policy_build":
            return self._activate(body)
        if path == "rpc/match_policy_chunks":
            if "match" in self.fail:
                return httpx.Response(500, json={"code": "XX000", "message": "boom"})
            return httpx.Response(200, json=self._match(body))
        return httpx.Response(404, json={"code": "PGRST205", "message": "not found"})


@pytest.fixture
def test_settings(tmp_path: Path, data_copy: Path) -> Settings:
    return make_settings(
        data_path=str(data_copy),
        chunk_size=500,
        chunk_overlap=100,
        retrieval_top_k=4,
        embedding_dimensions=KeywordEmbeddings.dim,
        **MOCK_SUPABASE,
    )


@pytest.fixture
def embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()


@pytest.fixture
def fake_db() -> FakePolicyDB:
    return FakePolicyDB()


@pytest.fixture
def store(test_settings: Settings, fake_db: FakePolicyDB) -> SupabasePolicyStore:
    return SupabasePolicyStore(test_settings, transport=httpx.MockTransport(fake_db.handler))

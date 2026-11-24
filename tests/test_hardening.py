"""End-to-end auth/RBAC/tenant/RAG/deployment checks through the real FastAPI stack.

Supabase is replaced by an ``httpx.MockTransport`` (no network); embeddings and the
LLM are offline fakes from conftest. No production credentials are used.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult

import ingest
import src.role_manager as role_manager
from ingest import run_ingestion
from main import app, get_pipeline
from src.config import Role
from src.rag_pipeline import RAGPipeline
from tests.conftest import FailingEmbeddings, RecordingChatModel, make_settings

# --- Mock Supabase ---------------------------------------------------------------------

TENANT_A, TENANT_B = "tenant-a", "tenant-b"
USERS: dict[str, dict[str, Any]] = {
    # token -> (user id, profile row)
    "tok.super.admin": {"id": "00000000-0000-4000-8000-000000000001", "role": "SUPER_ADMIN", "status": "Active", "tenant_id": TENANT_A},
    "tok.admin.a": {"id": "00000000-0000-4000-8000-000000000002", "role": "ADMIN", "status": "Active", "tenant_id": TENANT_A},
    "tok.admin.b": {"id": "00000000-0000-4000-8000-000000000003", "role": "ADMIN", "status": "Active", "tenant_id": TENANT_B},
    "tok.finance.a": {"id": "00000000-0000-4000-8000-000000000004", "role": "FINANCE_ANALYST", "status": "Active", "tenant_id": TENANT_A},
    "tok.support.a": {"id": "00000000-0000-4000-8000-000000000005", "role": "SUPPORT_EXECUTIVE", "status": "Active", "tenant_id": TENANT_A},
    "tok.support.b": {"id": "00000000-0000-4000-8000-000000000006", "role": "SUPPORT_EXECUTIVE", "status": "Active", "tenant_id": TENANT_B},
    "tok.customer.a": {"id": "00000000-0000-4000-8000-000000000007", "role": "CUSTOMER", "status": "Active", "tenant_id": TENANT_A},
    "tok.suspended.x": {"id": "00000000-0000-4000-8000-000000000008", "role": "ADMIN", "status": "Suspended", "tenant_id": TENANT_A},
}
EXPIRED = "tok.expired.x"


def supabase_handler(mode: str = "ok"):
    by_id = {u["id"]: u for u in USERS.values()}

    def handler(request: httpx.Request) -> httpx.Response:
        if mode == "down":
            raise httpx.ConnectError("down")
        if mode == "html":
            return httpx.Response(200, text="<html>Bad gateway</html>")
        if request.url.path == "/auth/v1/user":
            token = request.headers["authorization"].removeprefix("Bearer ")
            if token == EXPIRED:
                return httpx.Response(403, json={"code": 403, "error_code": "bad_jwt", "msg": "token is expired"})
            if token not in USERS:
                return httpx.Response(403, json={"code": 403, "error_code": "bad_jwt", "msg": "invalid JWT"})
            return httpx.Response(200, json={"id": USERS[token]["id"], "email": "x@example.com"})
        if request.url.path == "/rest/v1/profiles":
            user_id = request.url.params["id"].removeprefix("eq.")
            row = by_id.get(user_id)
            return httpx.Response(200, json=[{k: row[k] for k in ("role", "status", "tenant_id")}] if row else [])
        return httpx.Response(404)

    return handler


@pytest.fixture
def supabase(monkeypatch):
    """Route real auth code to the mock Supabase. Returns a function to switch modes."""
    settings = make_settings(
        supabase_url="https://mock-project.supabase.co", supabase_anon_key="anon", supabase_service_role_key="service"
    )
    state = {"mode": "ok"}
    original = role_manager.SupabaseAuthClient

    def factory(s, transport=None):
        return original(s, transport=httpx.MockTransport(supabase_handler(state["mode"])))

    monkeypatch.setattr(role_manager, "get_settings", lambda: settings)
    monkeypatch.setattr(role_manager, "SupabaseAuthClient", factory)
    return lambda mode: state.update(mode=mode)


def add_tenant_doc(settings, role_folder: str, tenant: str, name: str) -> None:
    tenant_dir = settings.data_dir / role_folder / tenant
    tenant_dir.mkdir()
    src = next((settings.data_dir / role_folder).glob("*.pdf"))
    (tenant_dir / name).write_bytes(src.read_bytes())


@pytest.fixture
def built_settings(test_settings, embeddings, store):
    add_tenant_doc(test_settings, "support_executive", TENANT_A, "tenant_a_confidential.pdf")
    outcomes = run_ingestion(test_settings, list(Role), embeddings=embeddings, store=store)
    assert all(o.status == "built" for o in outcomes), outcomes
    return test_settings


@pytest.fixture
def llm():
    return RecordingChatModel(response="Answer [complaint_handling_procedure.pdf, p. 1]")


@pytest.fixture
def client(built_settings, embeddings, llm, supabase, store):
    pipeline = RAGPipeline(built_settings, embeddings=embeddings, llm=llm, store=store)
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def status_call(client: TestClient, token: str | None, body: dict | None = None) -> httpx.Response:
    return client.post("/ingest/status", json=body or {}, headers=auth(token) if token else {})


# --- Authentication ----------------------------------------------------------------------

def test_missing_token_401_with_hint(client) -> None:
    r = status_call(client, None, {"roles": ["SUPER_ADMIN"]})
    assert r.status_code == 401
    assert r.json()["detail"].startswith("Missing bearer token") and "Authorize" in r.json()["detail"]
    assert r.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("token", ["not-a-jwt", "a.b.c", "tok.unknown.user"])
def test_invalid_token_401(client, token) -> None:
    assert status_call(client, token).status_code == 401


def test_expired_token_401(client) -> None:
    r = status_call(client, EXPIRED)
    assert r.status_code == 401 and r.json()["detail"] == "Invalid or expired access token"


def test_inactive_user_403(client) -> None:
    r = status_call(client, "tok.suspended.x")
    assert r.status_code == 403 and "inactive" in r.json()["detail"]


def test_supabase_outage_503(client, supabase) -> None:
    supabase("down")
    assert status_call(client, "tok.super.admin").status_code == 503
    supabase("html")
    assert status_call(client, "tok.super.admin").status_code == 503


def test_doubled_bearer_prefix_from_swagger_is_accepted(client) -> None:
    r = client.post("/ingest/status", json={}, headers={"Authorization": "Bearer Bearer tok.super.admin"})
    assert r.status_code == 200


# --- Authorization / admin endpoint ---------------------------------------------------

def test_super_admin_sees_global_metadata_and_last_ingest(client) -> None:
    body = status_call(client, "tok.super.admin").json()
    support = next(i for i in body["indexes"] if i["role"] == "SUPPORT_EXECUTIVE")
    assert "tenant_a_confidential.pdf" in support["documents"]
    assert support["tenants"] == ["GLOBAL", TENANT_A]
    assert body["last_ingest"]["roles"]["CUSTOMER"]["status"] == "built"
    assert "tenant_chunk_counts" not in support


def test_admin_sees_only_tenant_scoped_counts(client) -> None:
    full = {i["role"]: i for i in status_call(client, "tok.super.admin").json()["indexes"]}
    a = {i["role"]: i for i in status_call(client, "tok.admin.a").json()["indexes"]}
    b_resp = status_call(client, "tok.admin.b")
    b = {i["role"]: i for i in b_resp.json()["indexes"]}

    total = full["SUPPORT_EXECUTIVE"]["chunk_count"]
    assert b["SUPPORT_EXECUTIVE"]["chunk_count"] < a["SUPPORT_EXECUTIVE"]["chunk_count"] == total
    # Tenant B learns nothing about tenant A: no names, ids, totals or ingest report.
    assert TENANT_A not in b_resp.text and "tenant_a_confidential" not in b_resp.text
    assert b_resp.json()["last_ingest"] is None
    assert all(i["documents"] == [] and i["tenants"] == [] for i in b.values())


def test_admin_count_hidden_without_tenant_counts(client, fake_db) -> None:
    fake_db.indexes["SUPPORT_EXECUTIVE"]["tenant_chunk_counts"] = None
    item = status_call(client, "tok.admin.b", {"roles": ["SUPPORT_EXECUTIVE"]}).json()["indexes"][0]
    assert item["index_built"] and item["chunk_count"] is None  # never the all-tenant total


@pytest.mark.parametrize("token", ["tok.customer.a", "tok.finance.a", "tok.support.a"])
def test_non_admin_roles_denied_status(client, token) -> None:
    assert status_call(client, token).status_code == 403


def test_ingest_status_is_read_only(client, fake_db) -> None:
    fake_db.requests.clear()
    assert status_call(client, "tok.super.admin").status_code == 200
    assert fake_db.requests and all(method == "GET" for method, _ in fake_db.requests)
    for method in ("get", "put", "delete"):
        assert getattr(client, method)("/ingest/status", headers=auth("tok.super.admin")).status_code == 405


# --- RAG through the API, per role --------------------------------------------------------

@pytest.mark.parametrize(
    ("token", "role", "folder"),
    [
        ("tok.super.admin", "SUPER_ADMIN", "super_admin"),
        ("tok.admin.a", "ADMIN", "admin"),
        ("tok.finance.a", "FINANCE_ANALYST", "finance_analyst"),
        ("tok.support.a", "SUPPORT_EXECUTIVE", "support_executive"),
        ("tok.customer.a", "CUSTOMER", "customer"),
    ],
)
def test_chat_uses_only_own_role_documents(client, built_settings, llm, token, role, folder) -> None:
    r = client.post("/chat", json={"question": "What is the policy and procedure?"}, headers=auth(token))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["role"] == role
    own = {p.name for p in (built_settings.data_dir / folder).rglob("*.pdf")}
    assert set(body["source_documents"]) <= own
    # Nothing from another role's folder reached the LLM context.
    prompt = llm.calls[-1][1].content
    others = {p.name for p in built_settings.data_dir.rglob("*.pdf")} - own
    assert not any(name in prompt for name in others)


def test_cross_tenant_document_never_reaches_llm(client, llm) -> None:
    question = {"question": "complaint acknowledgement resolution target escalation"}
    r = client.post("/chat", json=question, headers=auth("tok.support.b"))
    assert r.status_code == 200
    assert "tenant_a_confidential.pdf" not in r.text
    assert "tenant_a_confidential.pdf" not in llm.calls[-1][1].content
    # Same question, same role, tenant A: their own document is retrievable.
    client.post("/chat", json=question, headers=auth("tok.support.a"))
    assert "tenant_a_confidential.pdf" in llm.calls[-1][1].content


def test_client_cannot_choose_tenant(client) -> None:
    r = client.post("/chat", json={"question": "complaint escalation", "tenant_id": TENANT_A}, headers=auth("tok.support.b"))
    assert r.status_code == 422


def _override_pipeline(pipeline: RAGPipeline) -> None:
    app.dependency_overrides[get_pipeline] = lambda: pipeline


def test_query_embedding_failure_is_502(client, built_settings, llm, store) -> None:
    _override_pipeline(RAGPipeline(built_settings, embeddings=FailingEmbeddings(), llm=llm, store=store))
    r = client.post("/chat", json={"question": "close my account"}, headers=auth("tok.customer.a"))
    assert r.status_code == 502 and "embedding" in r.json()["detail"]
    assert llm.calls == []


class ExplodingLLM(BaseChatModel):
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("provider 500")

    @property
    def _llm_type(self) -> str:
        return "exploding"


def test_llm_failure_is_502(client, built_settings, embeddings, store) -> None:
    _override_pipeline(RAGPipeline(built_settings, embeddings=embeddings, llm=ExplodingLLM(), store=store))
    r = client.post("/chat", json={"question": "close my account"}, headers=auth("tok.customer.a"))
    assert r.status_code == 502 and "language model" in r.json()["detail"]


def test_wrong_embedding_model_is_503(client, built_settings, embeddings, llm, store) -> None:
    other = built_settings.model_copy(update={"gemini_embedding_model": "text-embedding-004"})
    _override_pipeline(RAGPipeline(other, embeddings=embeddings, llm=llm, store=store))
    r = client.post("/chat", json={"question": "close my account"}, headers=auth("tok.customer.a"))
    assert r.status_code == 503 and llm.calls == []


@pytest.mark.parametrize("failure", ["down", "match", "missing"])
def test_policy_store_failure_is_503(client, built_settings, embeddings, llm, store, fake_db, failure) -> None:
    _override_pipeline(RAGPipeline(built_settings, embeddings=embeddings, llm=llm, store=store))
    fake_db.fail.add(failure)
    r = client.post("/chat", json={"question": "close my account"}, headers=auth("tok.customer.a"))
    assert r.status_code == 503 and llm.calls == []
    assert r.json() == {"detail": "The policy store is temporarily unavailable."}


def test_unexpected_error_is_500_without_stack_trace(client) -> None:
    class Broken:
        settings = None

        async def answer(self, *_: Any) -> None:
            raise RuntimeError("secret internal detail")

    app.dependency_overrides[get_pipeline] = lambda: Broken()
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/chat", json={"question": "close my account"}, headers=auth("tok.customer.a"))
    assert r.status_code == 500 and r.json() == {"detail": "Internal server error"}
    assert "secret" not in r.text and r.headers.get("x-request-id")


# --- Deployment / system --------------------------------------------------------------

def test_health_ok_when_all_ready_and_reveals_nothing_else(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": r.json()["version"], "indexes_built": 5, "indexes_total": 5}


def test_health_degraded_when_indexes_missing(test_settings, embeddings, store) -> None:
    _override_pipeline(RAGPipeline(test_settings, embeddings=embeddings, store=store))
    try:
        with TestClient(app) as c:
            body = c.get("/health").json()
    finally:
        app.dependency_overrides.clear()
    assert body["status"] == "degraded" and body["indexes_built"] == 0


def test_request_id_echoed_or_generated(client) -> None:
    assert client.get("/health", headers={"X-Request-ID": "abc-123"}).headers["x-request-id"] == "abc-123"
    generated = client.get("/health", headers={"X-Request-ID": "bad id\n!"}).headers["x-request-id"]
    assert generated != "bad id\n!" and len(generated) == 32


def test_openapi_bearer_security() -> None:
    spec = TestClient(app).get("/openapi.json").json()
    schemes = spec["components"]["securitySchemes"]
    (name, scheme), = schemes.items()
    assert scheme["type"] == "http" and scheme["scheme"] == "bearer" and scheme["bearerFormat"] == "JWT"
    assert spec["paths"]["/chat"]["post"]["security"] == [{name: []}]
    assert spec["paths"]["/ingest/status"]["post"]["security"] == [{name: []}]
    assert "security" not in spec["paths"]["/health"]["get"]
    assert "security" not in spec["paths"]["/roles"]["get"]


def test_health_degraded_when_supabase_down(client, fake_db) -> None:
    fake_db.fail.add("down")
    pipeline = app.dependency_overrides[get_pipeline]()
    pipeline._index_cache = None  # drop the 30 s cache
    body = client.get("/health").json()
    assert body["status"] == "degraded" and body["indexes_built"] == 0


def test_ingest_cli_configuration_error_exit_code(test_settings, monkeypatch, capsys) -> None:
    monkeypatch.setattr(ingest, "get_settings", lambda: test_settings)  # no GEMINI_API_KEY
    assert ingest.main([]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_ingest_failure_detail_is_redacted(test_settings, store, fake_db) -> None:
    class LeakyEmbeddings(FailingEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("PermissionDenied: API key AIzaSyA1234567890abcdefghijklmnopqrstu is invalid")

    outcome = run_ingestion(test_settings, [Role.ADMIN], embeddings=LeakyEmbeddings(), store=store)[0]
    stored = fake_db.indexes["ADMIN"]["last_run_detail"]
    assert outcome.status == "failed"
    assert "AIzaSy" not in outcome.detail and "AIzaSy" not in stored and "[redacted]" in stored


def test_index_row_records_tenant_counts(built_settings, fake_db) -> None:
    row = fake_db.indexes["SUPPORT_EXECUTIVE"]
    counts = row["tenant_chunk_counts"]
    assert set(counts) == {"GLOBAL", TENANT_A} and sum(counts.values()) == row["chunk_count"]


def test_llm_clients_have_bounded_timeout_and_retries() -> None:
    from src.chains import get_llm
    from src.config import LLMProvider

    settings = make_settings(gemini_api_key="dummy", groq_api_key="dummy", llm_timeout_seconds=12, llm_max_retries=1)
    gemini = get_llm(settings, LLMProvider.GEMINI)
    assert gemini.timeout == 12 and gemini.max_retries == 1
    groq = get_llm(settings, LLMProvider.GROQ)
    assert groq.request_timeout == 12 and groq.max_retries == 1


def test_llm_fallback_is_opt_in_and_validated() -> None:
    from langchain_core.runnables import RunnableWithFallbacks

    from src.chains import get_llm_with_fallback

    plain = get_llm_with_fallback(make_settings(gemini_api_key="dummy"))
    assert not isinstance(plain, RunnableWithFallbacks)
    wrapped = get_llm_with_fallback(make_settings(gemini_api_key="dummy", groq_api_key="dummy", llm_fallback_provider="GROQ"))
    assert isinstance(wrapped, RunnableWithFallbacks)
    assert make_settings(llm_fallback_provider="").llm_fallback_provider is None
    with pytest.raises(Exception, match="LLM_FALLBACK_PROVIDER"):
        make_settings(llm_provider="gemini", llm_fallback_provider="gemini")


def test_fallback_answers_when_primary_llm_fails(client, built_settings, embeddings, store) -> None:
    fallback = RecordingChatModel(response="Close via the app [account_terms_and_fees.pdf, p. 2]")
    _override_pipeline(RAGPipeline(built_settings, embeddings=embeddings, llm=ExplodingLLM().with_fallbacks([fallback]), store=store))
    r = client.post("/chat", json={"question": "close my account"}, headers=auth("tok.customer.a"))
    assert r.status_code == 200 and r.json()["found"] and fallback.calls

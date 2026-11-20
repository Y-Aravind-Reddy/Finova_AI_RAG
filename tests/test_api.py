from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ingest import run_ingestion
from main import app, get_pipeline
from src.config import Role
from src.rag_pipeline import RAGPipeline
from src.role_manager import AuthenticatedUser, get_current_user
from tests.conftest import RecordingChatModel


@pytest.fixture
def client(test_settings, embeddings):
    run_ingestion(test_settings, list(Role), embeddings=embeddings)
    pipeline = RAGPipeline(
        test_settings,
        embeddings=embeddings,
        llm=RecordingChatModel(response="Answer [account_terms_and_fees.pdf, p. 2]"),
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def as_user(role: Role, tenant: str | None = "bank-1") -> None:
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser("u1", role, tenant)


def test_health_and_roles(client) -> None:
    assert client.get("/health").json()["status"] == "ok"
    roles = [r["role"] for r in client.get("/roles").json()["roles"]]
    assert roles == ["SUPER_ADMIN", "ADMIN", "FINANCE_ANALYST", "SUPPORT_EXECUTIVE", "CUSTOMER"]


def test_chat_requires_token(client) -> None:
    response = client.post("/chat", json={"question": "How do I close my account?"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_chat_with_invalid_token_and_unconfigured_supabase(client) -> None:
    response = client.post(
        "/chat", json={"question": "hello there"}, headers={"Authorization": "Bearer a.b.c"}
    )
    assert response.status_code == 503  # Supabase not configured in tests -> fail closed


def test_chat_uses_verified_role(client) -> None:
    as_user(Role.CUSTOMER)
    response = client.post("/chat", json={"question": "How do I close my account?"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["role"] == "CUSTOMER" and body["found"] is True
    assert "account_terms_and_fees.pdf" in body["source_documents"]
    assert body["source_pages"] and all(isinstance(p, int) for p in body["source_pages"])


def test_client_supplied_role_is_rejected(client) -> None:
    as_user(Role.CUSTOMER)
    response = client.post("/chat", json={"question": "credit bands?", "role": "SUPER_ADMIN"})
    assert response.status_code == 422


def test_question_validation(client) -> None:
    as_user(Role.CUSTOMER)
    assert client.post("/chat", json={"question": "  "}).status_code == 422
    assert client.post("/chat", json={"question": "x" * 5000}).status_code == 422


def test_ingest_status_permissions(client) -> None:
    as_user(Role.CUSTOMER)
    assert client.post("/ingest/status").status_code == 403
    as_user(Role.ADMIN)
    admin_view = client.post("/ingest/status", json={"roles": ["CUSTOMER"]}).json()["indexes"]
    assert len(admin_view) == 1 and admin_view[0]["index_built"] and admin_view[0]["documents"] == []
    as_user(Role.SUPER_ADMIN)
    full = client.post("/ingest/status").json()["indexes"]
    assert len(full) == 5 and all(len(i["documents"]) == 2 for i in full)


def test_missing_index_returns_503(test_settings, embeddings) -> None:
    pipeline = RAGPipeline(test_settings, embeddings=embeddings, llm=RecordingChatModel())
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    as_user(Role.ADMIN)
    try:
        with TestClient(app) as c:
            response = c.post("/chat", json={"question": "offboarding steps"})
        assert response.status_code == 503
    finally:
        app.dependency_overrides.clear()

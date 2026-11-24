"""End-to-end ingestion + retrieval against the fake Supabase policy store (offline)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ingest import run_ingestion
from src.config import GLOBAL_TENANT_ID, ConfigurationError, Role
from src.prompts import NOT_FOUND_MESSAGE, NOT_FOUND_SENTINEL
from src.rag_pipeline import (
    IndexIncompatibleError,
    IndexNotFoundError,
    RAGPipeline,
    is_document_permitted,
    is_index_ready,
)
from src.role_manager import AuthenticatedUser
from src.vector_store import VectorStoreError, vector_literal
from tests.conftest import FailingEmbeddings, RecordingChatModel, make_settings

ROOT = Path(__file__).resolve().parent.parent


def user(role: Role, tenant: str | None = "bank-1") -> AuthenticatedUser:
    return AuthenticatedUser(user_id="u1", role=role, tenant_id=tenant)


@pytest.fixture
def built(test_settings, embeddings, store):
    outcomes = run_ingestion(test_settings, list(Role), embeddings=embeddings, store=store)
    assert [o.status for o in outcomes] == ["built"] * 5, outcomes
    return test_settings


def pipeline_for(settings, embeddings, store, **kwargs) -> RAGPipeline:
    return RAGPipeline(settings, embeddings=embeddings, store=store, **kwargs)


def test_ingestion_stores_five_roles(built, embeddings, store, fake_db) -> None:
    statuses = pipeline_for(built, embeddings, store).index_statuses(max_age=0)
    assert len(statuses) == 5 and all(is_index_ready(s) for s in statuses)
    for status in statuses:
        assert len(status["documents"]) == 2 and status["chunk_count"] > 0
        rows = [c for c in fake_db.chunks if c["role_scope"] == status["role"]]
        assert len(rows) == status["chunk_count"]
        assert {c["build_id"] for c in rows} == {fake_db.indexes[status["role"]]["active_build_id"]}
        assert all(c["allowed_roles"] == [status["role"]] for c in rows)
    assert all(row["last_run_status"] == "built" for row in fake_db.indexes.values())


def test_rerun_skips_and_rebuild_replaces_old_build(built, embeddings, store, fake_db) -> None:
    again = run_ingestion(built, [Role.CUSTOMER], embeddings=embeddings, store=store)
    assert again[0].status == "skipped"
    old_build = fake_db.indexes["CUSTOMER"]["active_build_id"]
    forced = run_ingestion(built, [Role.CUSTOMER], rebuild=True, embeddings=embeddings, store=store)
    assert forced[0].status == "built"
    new_build = fake_db.indexes["CUSTOMER"]["active_build_id"]
    assert new_build != old_build
    assert {c["build_id"] for c in fake_db.chunks if c["role_scope"] == "CUSTOMER"} == {new_build}


@pytest.mark.parametrize("failure", ["embed", "insert", "activate"])
def test_failed_rebuild_keeps_previous_build(built, embeddings, store, fake_db, failure) -> None:
    before = dict(fake_db.indexes["ADMIN"])
    chunks_before = [c for c in fake_db.chunks if c["role_scope"] == "ADMIN"]
    if failure == "embed":
        embeddings = FailingEmbeddings()
    else:
        fake_db.fail.add(failure)
    outcome = run_ingestion(built, [Role.ADMIN], rebuild=True, embeddings=embeddings, store=store)
    fake_db.fail.clear()
    assert outcome[0].status == "failed"
    after = fake_db.indexes["ADMIN"]
    assert after["active_build_id"] == before["active_build_id"]
    assert after["last_run_status"] == "failed"
    # Partial rows from the failed run were removed; the live build is intact.
    assert [c for c in fake_db.chunks if c["role_scope"] == "ADMIN"] == chunks_before


def test_empty_role_folder_fails_only_that_role(test_settings, embeddings, store) -> None:
    for pdf in (test_settings.data_dir / "admin").glob("*.pdf"):
        pdf.unlink()
    outcomes = {o.role: o for o in run_ingestion(test_settings, list(Role), embeddings=embeddings, store=store)}
    assert outcomes[Role.ADMIN].status == "failed" and "No PDF" in outcomes[Role.ADMIN].detail
    assert outcomes[Role.CUSTOMER].status == "built"


def test_dry_run_needs_no_embeddings_or_supabase(test_settings) -> None:
    settings = test_settings.model_copy(update={"supabase_url": None, "supabase_service_role_key": None})
    outcomes = run_ingestion(settings, list(Role), dry_run=True)
    assert all(o.status == "dry-run" for o in outcomes)


def test_ingestion_requires_supabase_before_embedding(test_settings) -> None:
    settings = test_settings.model_copy(update={"supabase_service_role_key": None})
    with pytest.raises(ConfigurationError, match="SUPABASE_SERVICE_ROLE_KEY"):
        run_ingestion(settings, [Role.ADMIN], embeddings=FailingEmbeddings())


def test_ingestion_reports_missing_migration(test_settings, embeddings, store, fake_db) -> None:
    fake_db.fail.add("missing")
    with pytest.raises(VectorStoreError, match="migration"):
        run_ingestion(test_settings, [Role.ADMIN], embeddings=embeddings, store=store)


def test_wrong_embedding_dimensions_fail_role(test_settings, embeddings, store, fake_db) -> None:
    settings = test_settings.model_copy(update={"embedding_dimensions": 3072})
    outcome = run_ingestion(settings, [Role.ADMIN], embeddings=embeddings, store=store)
    assert outcome[0].status == "failed" and "EMBEDDING_DIMENSIONS" in outcome[0].detail
    assert fake_db.chunks == []


def test_retrieval_is_role_scoped(built, embeddings, store) -> None:
    pipeline = pipeline_for(built, embeddings, store, llm=RecordingChatModel())
    hits = pipeline.retrieve("How do I close my account and transfer remaining funds?", user(Role.CUSTOMER))
    assert hits and all(d.metadata["role_scope"] == "CUSTOMER" for d, _ in hits)
    assert hits[0][0].metadata["source"] == "account_terms_and_fees.pdf"
    # A customer asking a finance question never sees finance documents.
    hits = pipeline.retrieve("internal credit score bands and debt-to-income ratio", user(Role.CUSTOMER))
    assert all(d.metadata["role_scope"] == "CUSTOMER" for d, _ in hits)
    # Even SUPER_ADMIN only searches its own role.
    hits = pipeline.retrieve("credit score bands debt-to-income", user(Role.SUPER_ADMIN))
    assert all(d.metadata["role_scope"] == "SUPER_ADMIN" for d, _ in hits)


def test_answer_has_citations_from_metadata(built, embeddings, store) -> None:
    llm = RecordingChatModel(response="Submit a closure request [account_terms_and_fees.pdf, p. 2]")
    result = asyncio.run(pipeline_for(built, embeddings, store, llm=llm).answer("How do I close my account?", user(Role.CUSTOMER)))
    assert result.found and result.role == Role.CUSTOMER
    assert any(s.document == "account_terms_and_fees.pdf" and s.page == 2 for s in result.sources)
    system, human = llm.calls[0]
    assert "CUSTOMER" in system.content and "untrusted DATA" in system.content
    assert "<documents>" in human.content and "credit_risk" not in human.content


def test_sources_narrowed_to_cited_and_unforgeable(built, embeddings, store) -> None:
    llm = RecordingChatModel(
        response="Close it in the app [source=account_terms_and_fees.pdf, p. 2] [credit_risk_assessment_guidelines.pdf, p. 1]"
    )
    result = asyncio.run(pipeline_for(built, embeddings, store, llm=llm).answer("How do I close my account?", user(Role.CUSTOMER)))
    # Only the cited, retrieved chunk is kept; the forged finance citation is ignored.
    assert [(s.document, s.page) for s in result.sources] == [("account_terms_and_fees.pdf", 2)]


def test_uncited_answer_falls_back_to_retrieved(built, embeddings, store) -> None:
    llm = RecordingChatModel(response="Submit a closure request.")
    result = asyncio.run(pipeline_for(built, embeddings, store, llm=llm).answer("How do I close my account?", user(Role.CUSTOMER)))
    assert len(result.sources) > 1


def test_not_found_sentinel(built, embeddings, store) -> None:
    pipeline = pipeline_for(built, embeddings, store, llm=RecordingChatModel(response=NOT_FOUND_SENTINEL))
    result = asyncio.run(pipeline.answer("What is the weather on Mars?", user(Role.CUSTOMER)))
    assert not result.found and result.answer == NOT_FOUND_MESSAGE and result.sources == []


def test_no_hits_skips_llm(built, embeddings, store) -> None:
    settings = built.model_copy(update={"retrieval_score_threshold": 0.99})
    llm = RecordingChatModel()
    result = asyncio.run(pipeline_for(settings, embeddings, store, llm=llm).answer("zzzz qqqq", user(Role.ADMIN)))
    assert not result.found and llm.calls == []


def test_tenant_isolation(test_settings, embeddings, store, fake_db) -> None:
    tenant_dir = test_settings.data_dir / "support_executive" / "tenant-a"
    tenant_dir.mkdir()
    # Reuse a sample PDF as tenant-a's confidential document.
    src = test_settings.data_dir / "support_executive" / "complaint_handling_procedure.pdf"
    (tenant_dir / "tenant_a_confidential.pdf").write_bytes(src.read_bytes())
    run_ingestion(test_settings, [Role.SUPPORT_EXECUTIVE], embeddings=embeddings, store=store)
    pipeline = pipeline_for(test_settings.model_copy(update={"retrieval_top_k": 10}), embeddings, store)

    q = "complaint acknowledgement resolution target escalation"
    a_sources = {d.metadata["source"] for d, _ in pipeline.retrieve(q, user(Role.SUPPORT_EXECUTIVE, "tenant-a"))}
    b_sources = {d.metadata["source"] for d, _ in pipeline.retrieve(q, user(Role.SUPPORT_EXECUTIVE, "tenant-b"))}
    none_sources = {d.metadata["source"] for d, _ in pipeline.retrieve(q, user(Role.SUPPORT_EXECUTIVE, None))}
    assert "tenant_a_confidential.pdf" in a_sources
    assert "tenant_a_confidential.pdf" not in b_sources
    assert "tenant_a_confidential.pdf" not in none_sources
    assert "complaint_handling_procedure.pdf" in b_sources  # GLOBAL docs remain visible
    assert fake_db.indexes["SUPPORT_EXECUTIVE"]["tenant_chunk_counts"]["tenant-a"] > 0


def test_api_drops_rows_the_database_should_not_have_returned(built, embeddings, store, fake_db, monkeypatch) -> None:
    """Defence in depth: even if the SQL filter were wrong, the API re-checks each row."""
    leaked = dict(next(c for c in fake_db.chunks if c["role_scope"] == "CUSTOMER"))
    leaked.update(tenant_id="tenant-b", chunk_id="leak", similarity=1.0)
    leaked.pop("embedding")
    monkeypatch.setattr(store, "match", lambda *a, **k: [leaked])
    hits = pipeline_for(built, embeddings, store).retrieve("anything", user(Role.CUSTOMER, "tenant-a"))
    assert hits == []


def test_permission_check_denies_missing_metadata() -> None:
    u = user(Role.ADMIN, "t1")
    good = {"role_scope": "ADMIN", "allowed_roles": ["ADMIN"], "tenant_id": GLOBAL_TENANT_ID}
    assert is_document_permitted(good, u)
    assert not is_document_permitted({**good, "tenant_id": "t2"}, u)
    assert not is_document_permitted({**good, "allowed_roles": ["CUSTOMER"]}, u)
    assert not is_document_permitted({**good, "role_scope": "SUPER_ADMIN"}, u)
    assert not is_document_permitted({"role_scope": "ADMIN"}, u)


def test_missing_and_incompatible_index(test_settings, embeddings, store) -> None:
    with pytest.raises(IndexNotFoundError):
        pipeline_for(test_settings, embeddings, store).retrieve("anything", user(Role.ADMIN))
    run_ingestion(test_settings, [Role.ADMIN], embeddings=embeddings, store=store)
    other = test_settings.model_copy(update={"gemini_embedding_model": "text-embedding-004"})
    with pytest.raises(IndexIncompatibleError):
        pipeline_for(other, embeddings, store).retrieve("anything", user(Role.ADMIN))


def test_supabase_outage_raises_vector_store_error(built, embeddings, store, fake_db) -> None:
    fake_db.fail.add("down")
    with pytest.raises(VectorStoreError):
        pipeline_for(built, embeddings, store).retrieve("close my account", user(Role.CUSTOMER))


def test_vector_literal_is_pgvector_text() -> None:
    assert vector_literal([0.5, -1, 2.0e-10]) == "[0.5,-1,2e-10]"


def test_local_vectorstore_folder_is_gone() -> None:
    assert not (ROOT / "vectorstore").exists()
    assert (ROOT / "supabase" / "migrations").is_dir()


def test_default_dimensions_match_migration() -> None:
    sql = next((ROOT / "supabase" / "migrations").glob("*_policy_vector_store.sql")).read_text(encoding="utf-8")
    dims = make_settings().embedding_dimensions
    assert sql.count(f"vector({dims})") == 2  # column + function parameter

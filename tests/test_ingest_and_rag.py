"""End-to-end ingestion + retrieval with offline fake embeddings and LLM."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ingest import run_ingestion
from src.config import GLOBAL_TENANT_ID, Role
from src.prompts import NOT_FOUND_MESSAGE, NOT_FOUND_SENTINEL
from src.rag_pipeline import (
    IndexIncompatibleError,
    IndexNotFoundError,
    RAGPipeline,
    index_status,
    is_document_permitted,
)
from src.role_manager import AuthenticatedUser
from tests.conftest import FailingEmbeddings, RecordingChatModel


def user(role: Role, tenant: str | None = "bank-1") -> AuthenticatedUser:
    return AuthenticatedUser(user_id="u1", role=role, tenant_id=tenant)


@pytest.fixture
def built(test_settings, embeddings):
    outcomes = run_ingestion(test_settings, list(Role), embeddings=embeddings)
    assert [o.status for o in outcomes] == ["built"] * 5, outcomes
    return test_settings


def test_ingestion_creates_five_indexes(built) -> None:
    for role in Role:
        status = index_status(role, built)
        assert status["index_built"] and status["embedding_model_matches"]
        assert len(status["documents"]) == 2
        index_dir = built.vectorstore_dir / role.folder_name
        manifest = json.loads((index_dir / "manifest.json").read_text())
        assert manifest["role"] == role.value and manifest["chunk_count"] > 0
        assert (index_dir / "index.faiss").is_file() and (index_dir / ".gitkeep").is_file()
    # No temporary build/backup directories left behind.
    assert sorted(p.name for p in built.vectorstore_dir.iterdir()) == sorted(r.folder_name for r in Role)


def test_rerun_skips_and_rebuild_rebuilds(built, embeddings) -> None:
    again = run_ingestion(built, [Role.CUSTOMER], embeddings=embeddings)
    assert again[0].status == "skipped"
    forced = run_ingestion(built, [Role.CUSTOMER], rebuild=True, embeddings=embeddings)
    assert forced[0].status == "built"


def test_failed_rebuild_keeps_previous_index(built) -> None:
    before = index_status(Role.ADMIN, built)
    outcome = run_ingestion(built, [Role.ADMIN], rebuild=True, embeddings=FailingEmbeddings())
    assert outcome[0].status == "failed"
    after = index_status(Role.ADMIN, built)
    assert after["index_built"] and after["built_at"] == before["built_at"]


def test_empty_role_folder_fails_only_that_role(test_settings, embeddings) -> None:
    for pdf in (test_settings.data_dir / "admin").glob("*.pdf"):
        pdf.unlink()
    outcomes = {o.role: o for o in run_ingestion(test_settings, list(Role), embeddings=embeddings)}
    assert outcomes[Role.ADMIN].status == "failed" and "No PDF" in outcomes[Role.ADMIN].detail
    assert outcomes[Role.CUSTOMER].status == "built"


def test_dry_run_needs_no_embeddings(test_settings) -> None:
    outcomes = run_ingestion(test_settings, list(Role), dry_run=True)
    assert all(o.status == "dry-run" for o in outcomes)
    assert not test_settings.vectorstore_dir.exists()


def test_retrieval_is_role_scoped(built, embeddings) -> None:
    pipeline = RAGPipeline(built, embeddings=embeddings, llm=RecordingChatModel())
    hits = pipeline.retrieve("How do I close my account and transfer remaining funds?", user(Role.CUSTOMER))
    assert hits and all(d.metadata["role_scope"] == "CUSTOMER" for d, _ in hits)
    assert hits[0][0].metadata["source"] == "account_terms_and_fees.pdf"
    # A customer asking a finance question never sees finance documents.
    hits = pipeline.retrieve("internal credit score bands and debt-to-income ratio", user(Role.CUSTOMER))
    assert all(d.metadata["role_scope"] == "CUSTOMER" for d, _ in hits)
    # Even SUPER_ADMIN only searches its own index.
    hits = pipeline.retrieve("credit score bands debt-to-income", user(Role.SUPER_ADMIN))
    assert all(d.metadata["role_scope"] == "SUPER_ADMIN" for d, _ in hits)


def test_answer_has_citations_from_metadata(built, embeddings) -> None:
    llm = RecordingChatModel(response="Submit a closure request [account_terms_and_fees.pdf, p. 2]")
    pipeline = RAGPipeline(built, embeddings=embeddings, llm=llm)
    result = asyncio.run(pipeline.answer("How do I close my account?", user(Role.CUSTOMER)))
    assert result.found and result.role == Role.CUSTOMER
    assert any(s.document == "account_terms_and_fees.pdf" and s.page == 2 for s in result.sources)
    system, human = llm.calls[0]
    assert "CUSTOMER" in system.content and "untrusted DATA" in system.content
    assert "<documents>" in human.content and "credit_risk" not in human.content


def test_sources_narrowed_to_cited_and_unforgeable(built, embeddings) -> None:
    llm = RecordingChatModel(
        response="Close it in the app [source=account_terms_and_fees.pdf, p. 2] [credit_risk_assessment_guidelines.pdf, p. 1]"
    )
    result = asyncio.run(RAGPipeline(built, embeddings=embeddings, llm=llm).answer("How do I close my account?", user(Role.CUSTOMER)))
    # Only the cited, retrieved chunk is kept; the forged finance citation is ignored.
    assert [(s.document, s.page) for s in result.sources] == [("account_terms_and_fees.pdf", 2)]


def test_uncited_answer_falls_back_to_retrieved(built, embeddings) -> None:
    llm = RecordingChatModel(response="Submit a closure request.")
    result = asyncio.run(RAGPipeline(built, embeddings=embeddings, llm=llm).answer("How do I close my account?", user(Role.CUSTOMER)))
    assert len(result.sources) > 1


def test_not_found_sentinel(built, embeddings) -> None:
    pipeline = RAGPipeline(built, embeddings=embeddings, llm=RecordingChatModel(response=NOT_FOUND_SENTINEL))
    result = asyncio.run(pipeline.answer("What is the weather on Mars?", user(Role.CUSTOMER)))
    assert not result.found and result.answer == NOT_FOUND_MESSAGE and result.sources == []


def test_no_hits_skips_llm(built, embeddings) -> None:
    settings = built.model_copy(update={"retrieval_score_threshold": 0.99})
    llm = RecordingChatModel()
    result = asyncio.run(RAGPipeline(settings, embeddings=embeddings, llm=llm).answer("zzzz qqqq", user(Role.ADMIN)))
    assert not result.found and llm.calls == []


def test_tenant_isolation(test_settings, embeddings) -> None:
    tenant_dir = test_settings.data_dir / "support_executive" / "tenant-a"
    tenant_dir.mkdir()
    # Reuse a sample PDF as tenant-a's confidential document.
    src = test_settings.data_dir / "support_executive" / "complaint_handling_procedure.pdf"
    (tenant_dir / "tenant_a_confidential.pdf").write_bytes(src.read_bytes())
    run_ingestion(test_settings, [Role.SUPPORT_EXECUTIVE], embeddings=embeddings)
    pipeline = RAGPipeline(test_settings.model_copy(update={"retrieval_top_k": 10}), embeddings=embeddings)

    q = "complaint acknowledgement resolution target escalation"
    a_sources = {d.metadata["source"] for d, _ in pipeline.retrieve(q, user(Role.SUPPORT_EXECUTIVE, "tenant-a"))}
    b_sources = {d.metadata["source"] for d, _ in pipeline.retrieve(q, user(Role.SUPPORT_EXECUTIVE, "tenant-b"))}
    none_sources = {d.metadata["source"] for d, _ in pipeline.retrieve(q, user(Role.SUPPORT_EXECUTIVE, None))}
    assert "tenant_a_confidential.pdf" in a_sources
    assert "tenant_a_confidential.pdf" not in b_sources
    assert "tenant_a_confidential.pdf" not in none_sources
    assert "complaint_handling_procedure.pdf" in b_sources  # GLOBAL docs remain visible


def test_permission_check_denies_missing_metadata() -> None:
    u = user(Role.ADMIN, "t1")
    good = {"role_scope": "ADMIN", "allowed_roles": ["ADMIN"], "tenant_id": GLOBAL_TENANT_ID}
    assert is_document_permitted(good, u)
    assert not is_document_permitted({**good, "tenant_id": "t2"}, u)
    assert not is_document_permitted({**good, "allowed_roles": ["CUSTOMER"]}, u)
    assert not is_document_permitted({**good, "role_scope": "SUPER_ADMIN"}, u)
    assert not is_document_permitted({"role_scope": "ADMIN"}, u)


def test_missing_and_incompatible_index(test_settings, embeddings) -> None:
    pipeline = RAGPipeline(test_settings, embeddings=embeddings)
    with pytest.raises(IndexNotFoundError):
        pipeline.retrieve("anything", user(Role.ADMIN))
    run_ingestion(test_settings, [Role.ADMIN], embeddings=embeddings)
    other = test_settings.model_copy(update={"gemini_embedding_model": "text-embedding-004"})
    with pytest.raises(IndexIncompatibleError):
        RAGPipeline(other, embeddings=embeddings).retrieve("anything", user(Role.ADMIN))


def test_vectorstore_folders_exist_in_project() -> None:
    root = Path(__file__).resolve().parent.parent / "vectorstore"
    for role in Role:
        assert (root / role.folder_name).is_dir()

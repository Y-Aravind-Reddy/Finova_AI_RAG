from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from src.config import Role
from src.role_manager import (
    AuthenticationError,
    AuthorizationError,
    AuthServiceUnavailable,
    SupabaseAuthClient,
    authenticate_demo,
    authenticate_supabase,
    authenticate_token,
)
from tests.conftest import make_settings

USER_ID = "6f1c2a4e-1111-4b2b-9a77-1234567890ab"
TOKEN = "header.payload.signature"


def supabase_settings() -> Any:
    return make_settings(
        supabase_url="https://demo-project.supabase.co",
        supabase_anon_key="anon",
        supabase_service_role_key="service",
    )


def mock_client(
    user_status: int = 200,
    profile: list[dict] | None = None,
    profile_status: int = 200,
    seen: list[httpx.Request] | None = None,
) -> SupabaseAuthClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if request.url.path == "/auth/v1/user":
            if user_status != 200:
                return httpx.Response(user_status, json={"msg": "invalid JWT"})
            return httpx.Response(200, json={"id": USER_ID, "email": "user@example.com"})
        if request.url.path == "/rest/v1/profiles":
            return httpx.Response(profile_status, json=profile if profile is not None else [])
        return httpx.Response(404)

    return SupabaseAuthClient(supabase_settings(), transport=httpx.MockTransport(handler))


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_valid_token_returns_role_from_profile() -> None:
    seen: list[httpx.Request] = []
    client = mock_client(profile=[{"role": "finance_analyst", "is_active": True, "tenant_id": "bank-1"}], seen=seen)
    user = run(authenticate_supabase(TOKEN, supabase_settings(), client))
    assert user.role == Role.FINANCE_ANALYST
    assert user.tenant_id == "bank-1" and user.user_id == USER_ID
    # Token verified with anon key; profile read filtered by the *verified* user id.
    assert seen[0].headers["authorization"] == f"Bearer {TOKEN}"
    assert seen[1].url.params["id"] == f"eq.{USER_ID}"


def test_expired_or_invalid_token_rejected() -> None:
    with pytest.raises(AuthenticationError):
        run(authenticate_supabase(TOKEN, supabase_settings(), mock_client(user_status=401)))


def test_malformed_token_rejected_without_network() -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(AuthenticationError):
        run(authenticate_supabase("not-a-jwt", supabase_settings(), mock_client(seen=seen)))
    assert seen == []


def test_inactive_user_rejected() -> None:
    client = mock_client(profile=[{"role": "ADMIN", "is_active": False, "tenant_id": "t"}])
    with pytest.raises(AuthorizationError, match="inactive"):
        run(authenticate_supabase(TOKEN, supabase_settings(), client))


def test_missing_is_active_treated_as_inactive() -> None:
    client = mock_client(profile=[{"role": "ADMIN", "tenant_id": "t"}])
    with pytest.raises(AuthorizationError):
        run(authenticate_supabase(TOKEN, supabase_settings(), client))


def test_missing_profile_rejected() -> None:
    with pytest.raises(AuthorizationError, match="profile"):
        run(authenticate_supabase(TOKEN, supabase_settings(), mock_client(profile=[])))


def test_unknown_role_rejected() -> None:
    client = mock_client(profile=[{"role": "AUDITOR", "is_active": True}])
    with pytest.raises(AuthorizationError):
        run(authenticate_supabase(TOKEN, supabase_settings(), client))


def test_supabase_down_is_503() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = SupabaseAuthClient(supabase_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(AuthServiceUnavailable):
        run(authenticate_supabase(TOKEN, supabase_settings(), client))


def test_unconfigured_supabase_is_503() -> None:
    with pytest.raises(AuthServiceUnavailable):
        run(authenticate_token(TOKEN, make_settings()))


def test_demo_tokens_only_when_enabled() -> None:
    dev = make_settings(auth_mode="demo", allow_demo_auth=True, app_env="development")
    user = authenticate_demo("demo:support_executive:bank-9", dev)
    assert user.role == Role.SUPPORT_EXECUTIVE and user.tenant_id == "bank-9" and user.is_demo
    with pytest.raises(AuthenticationError):
        authenticate_demo("demo:ROOT", dev)
    with pytest.raises(AuthenticationError):
        authenticate_demo("demo:ADMIN", make_settings())  # supabase mode: demo disabled
    # In supabase mode a demo-looking token is just an invalid JWT.
    with pytest.raises(AuthenticationError):
        run(authenticate_token("demo:SUPER_ADMIN", supabase_settings()))

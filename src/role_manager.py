"""Supabase access-token validation and role authorization.

Trust model:
* The frontend sends ONLY the Supabase access token (``Authorization: Bearer``).
* The backend validates the token with Supabase Auth (``GET /auth/v1/user``),
  which rejects expired, malformed, revoked and forged tokens.
* The role, status and tenant are read from the ``profiles`` table using
  the verified user id. Any role/tenant sent by the client is ignored.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import httpx
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import AuthMode, ConfigurationError, Role, Settings, get_settings

logger = logging.getLogger(__name__)

# Which role indexes each role may search. Deliberately one-to-one: no role,
# including SUPER_ADMIN, can read another role's restricted documents. Extend
# only with an explicit, reviewed policy decision.
ROLE_INDEX_ACCESS: dict[Role, tuple[Role, ...]] = {role: (role,) for role in Role}


class AuthError(Exception):
    """Base class for auth failures; carries the HTTP status to return."""

    status_code: int = 401

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class AuthenticationError(AuthError):
    """Missing, invalid or expired credentials (HTTP 401)."""

    status_code = 401


class AuthorizationError(AuthError):
    """Authenticated but not permitted: inactive, no profile, wrong role (HTTP 403)."""

    status_code = 403


class AuthServiceUnavailable(AuthError):
    """Supabase unreachable or auth not configured (HTTP 503)."""

    status_code = 503


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    role: Role
    tenant_id: str | None
    email: str | None = None
    is_demo: bool = False

    @property
    def accessible_roles(self) -> tuple[Role, ...]:
        return ROLE_INDEX_ACCESS[self.role]


def parse_role(value: object) -> Role:
    """Map a stored profile role to ``Role``; unknown values are rejected."""
    if not isinstance(value, str):
        raise AuthorizationError("User profile has no valid role")
    try:
        return Role(value.strip().upper())
    except ValueError as exc:
        raise AuthorizationError("User profile role is not recognised") from exc


def is_profile_active(profile: dict) -> bool:
    """Fail closed: only an explicit ``status`` of "active" (any case) is accepted."""
    status = profile.get("status")
    return isinstance(status, str) and status.strip().lower() == "active"


def _looks_like_jwt(token: str) -> bool:
    parts = token.split(".")
    return len(parts) == 3 and all(parts) and len(token) < 8192


class SupabaseAuthClient:
    """Thin async client for the two Supabase calls the backend needs."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._settings = settings
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._settings.supabase_timeout_seconds,
            transport=self._transport,
        )

    async def get_user(self, token: str) -> dict:
        """Validate ``token`` with Supabase Auth and return the user object."""
        url, anon_key, _ = self._settings.require_supabase()
        try:
            async with self._client() as client:
                response = await client.get(
                    f"{url}/auth/v1/user",
                    headers={"apikey": anon_key, "Authorization": f"Bearer {token}"},
                )
        except httpx.HTTPError as exc:
            logger.error("Supabase auth request failed: %s", type(exc).__name__)
            raise AuthServiceUnavailable("Authentication service unavailable") from exc

        if response.status_code in (401, 403):
            raise AuthenticationError("Invalid or expired access token")
        if response.status_code != 200:
            logger.error("Unexpected Supabase auth status %s", response.status_code)
            raise AuthServiceUnavailable("Authentication service error")

        data = response.json()
        if not isinstance(data, dict) or not data.get("id"):
            raise AuthenticationError("Invalid or expired access token")
        return data

    async def get_profile(self, user_id: str) -> dict | None:
        """Fetch role/status/tenant_id for ``user_id`` using the service role key."""
        url, _, service_key = self._settings.require_supabase()
        table = self._settings.supabase_profiles_table
        try:
            async with self._client() as client:
                response = await client.get(
                    f"{url}/rest/v1/{table}",
                    params={"id": f"eq.{user_id}", "select": "role,status,tenant_id", "limit": "1"},
                    headers={
                        "apikey": service_key,
                        "Authorization": f"Bearer {service_key}",
                        "Accept": "application/json",
                    },
                )
        except httpx.HTTPError as exc:
            logger.error("Supabase profile request failed: %s", type(exc).__name__)
            raise AuthServiceUnavailable("Profile service unavailable") from exc

        if response.status_code != 200:
            logger.error("Unexpected Supabase profile status %s", response.status_code)
            raise AuthServiceUnavailable("Profile service error")
        rows = response.json()
        if not isinstance(rows, list) or not rows:
            return None
        return rows[0]


async def authenticate_supabase(
    token: str, settings: Settings, client: SupabaseAuthClient | None = None
) -> AuthenticatedUser:
    if not _looks_like_jwt(token):
        raise AuthenticationError("Invalid or expired access token")
    client = client or SupabaseAuthClient(settings)

    user = await client.get_user(token)
    user_id = str(user["id"])
    try:
        user_id = str(uuid.UUID(user_id))  # also guards the REST filter value
    except ValueError as exc:
        raise AuthenticationError("Invalid user id in token") from exc

    profile = await client.get_profile(user_id)
    if profile is None:
        raise AuthorizationError("No profile found for this user")
    if not is_profile_active(profile):
        raise AuthorizationError("User account is inactive")

    role = parse_role(profile.get("role"))
    tenant = profile.get("tenant_id")
    tenant_id = str(tenant) if tenant not in (None, "") else None
    return AuthenticatedUser(user_id=user_id, role=role, tenant_id=tenant_id, email=user.get("email"))


def authenticate_demo(token: str, settings: Settings) -> AuthenticatedUser:
    """Development-only auth. Token format: ``demo:<ROLE>[:<tenant_id>]``."""
    if not settings.demo_auth_enabled:  # defence in depth; config also forbids it
        raise AuthenticationError("Demo authentication is disabled")
    parts = token.split(":")
    if len(parts) not in (2, 3) or parts[0] != "demo":
        raise AuthenticationError("Invalid demo token. Use demo:<ROLE>[:<tenant_id>]")
    try:
        role = Role(parts[1].upper())
    except ValueError as exc:
        raise AuthenticationError("Unknown role in demo token") from exc
    tenant_id = parts[2] if len(parts) == 3 and parts[2] else None
    return AuthenticatedUser(
        user_id=f"demo-{role.folder_name}", role=role, tenant_id=tenant_id, is_demo=True
    )


async def authenticate_token(token: str, settings: Settings | None = None) -> AuthenticatedUser:
    """Validate a bearer token according to AUTH_MODE."""
    settings = settings or get_settings()
    token = token.strip()
    if not token:
        raise AuthenticationError("Missing access token")
    if settings.auth_mode == AuthMode.DEMO:
        return authenticate_demo(token, settings)
    try:
        return await authenticate_supabase(token, settings)
    except ConfigurationError as exc:
        logger.error("Supabase auth misconfigured: %s", exc)
        raise AuthServiceUnavailable("Authentication is not configured on the server") from exc


# --- FastAPI dependencies -----------------------------------------------------

_bearer = HTTPBearer(auto_error=False, description="Supabase access token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationError("Missing bearer token")
    return await authenticate_token(credentials.credentials)


def require_roles(allowed: Iterable[Role]) -> Callable[..., object]:
    """Dependency factory restricting an endpoint to ``allowed`` roles."""
    allowed_set = frozenset(allowed)

    async def _dependency(user: AuthenticatedUser = Depends(get_current_user)) -> AuthenticatedUser:
        if user.role not in allowed_set:
            raise AuthorizationError("Your role is not permitted to use this endpoint")
        return user

    return _dependency

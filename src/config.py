"""Environment configuration and validation.

All settings are read from environment variables (and the project ``.env`` file).
Structural validation (enum values, numeric ranges, unsafe combinations) happens
at load time so a misconfigured server fails fast. Secrets such as API keys are
validated lazily via the ``require_*`` helpers, so that endpoints which do not
need a key (e.g. ``/health``) still work while the operator is setting up.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the directory containing main.py / ingest.py.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Role(str, Enum):
    """The only roles recognised by the system. Do not add roles here casually:
    every role needs its own data folder, profile value and an allowed value in the
    policy_indexes/policy_chunks CHECK constraints (supabase/migrations/)."""

    SUPER_ADMIN = "SUPER_ADMIN"
    ADMIN = "ADMIN"
    FINANCE_ANALYST = "FINANCE_ANALYST"
    SUPPORT_EXECUTIVE = "SUPPORT_EXECUTIVE"
    CUSTOMER = "CUSTOMER"

    @property
    def folder_name(self) -> str:
        """Folder name used under data/ (e.g. ``finance_analyst``)."""
        return self.value.lower()


ROLE_DESCRIPTIONS: dict[Role, str] = {
    Role.SUPER_ADMIN: "Platform owner: governance, platform-wide security and incident policies.",
    Role.ADMIN: "Tenant/bank administrator: user administration and operational controls.",
    Role.FINANCE_ANALYST: "Finance team: credit, treasury and reporting policies.",
    Role.SUPPORT_EXECUTIVE: "Customer support: complaint handling and service procedures.",
    Role.CUSTOMER: "End customer: published account and service terms.",
}

# Tenant id stamped on documents that are shared by every tenant (e.g. generic
# platform policies placed directly in data/<role>/).
GLOBAL_TENANT_ID = "GLOBAL"


class LLMProvider(str, Enum):
    GEMINI = "gemini"
    GROQ = "groq"


class EmbeddingProvider(str, Enum):
    GEMINI = "gemini"


class AuthMode(str, Enum):
    SUPABASE = "supabase"
    DEMO = "demo"


class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class ConfigurationError(RuntimeError):
    """Raised when a required setting is missing or invalid."""


class Settings(BaseSettings):
    """Typed application settings."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Environment -------------------------------------------------------
    # Defaults to production so that forgetting to set it can never enable
    # development-only behaviour such as demo authentication.
    app_env: AppEnv = AppEnv.PRODUCTION

    # --- LLM / embeddings ----------------------------------------------------
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    # Must match the vector(N) column in the Supabase migration (3072 = Gemini default).
    embedding_dimensions: int = Field(default=3072, ge=8, le=16000)
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-120b"
    llm_provider: LLMProvider = LLMProvider.GEMINI
    # Optional second provider tried when the primary fails (e.g. Gemini 503 overload).
    llm_fallback_provider: LLMProvider | None = None
    embedding_provider: EmbeddingProvider = EmbeddingProvider.GEMINI
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    # Provider SDK defaults are "no timeout" and up to 6 retries, which can hang
    # /chat for minutes; bound them so failures surface quickly as HTTP 502.
    llm_timeout_seconds: float = Field(default=45.0, gt=0, le=300)
    llm_max_retries: int = Field(default=2, ge=0, le=6)

    # --- Supabase ------------------------------------------------------------
    supabase_url: str | None = None
    supabase_anon_key: SecretStr | None = None
    supabase_service_role_key: SecretStr | None = None
    supabase_profiles_table: str = "profiles"
    supabase_timeout_seconds: float = Field(default=10.0, gt=0, le=60)

    # --- Chunking / retrieval -----------------------------------------------
    chunk_size: int = Field(default=1000, ge=100, le=8000)
    chunk_overlap: int = Field(default=200, ge=0, le=4000)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    # Minimum relevance score in [0, 1]; 0 disables the threshold.
    retrieval_score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    max_question_chars: int = Field(default=2000, ge=10, le=10000)

    # --- Auth ----------------------------------------------------------------
    auth_mode: AuthMode = AuthMode.SUPABASE
    allow_demo_auth: bool = False

    # --- Paths / HTTP --------------------------------------------------------
    # Policy chunks and embeddings live in Supabase (see supabase/migrations/).
    data_path: str = "data"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"
    # Optional regex for origins that change per deployment/preview, e.g.
    # Google AI Studio / Cloud Run: ^https://[a-z0-9-]+\.(run\.app|usercontent\.goog)$
    cors_origin_regex: str | None = None

    # ------------------------------------------------------------------------
    @field_validator(
        "gemini_api_key",
        "groq_api_key",
        "supabase_anon_key",
        "supabase_service_role_key",
        mode="before",
    )
    @classmethod
    def _blank_secret_is_none(cls, value: object) -> object:
        # A blank line such as ``GEMINI_API_KEY=`` in .env means "not set".
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("cors_origin_regex", mode="after")
    @classmethod
    def _valid_regex(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip()
        if value in (".*", "^.*$", ".+", "^.+$"):
            raise ValueError("CORS_ORIGIN_REGEX must not allow every origin")
        if not (value.startswith("^") and value.endswith("$")):
            raise ValueError("CORS_ORIGIN_REGEX must be anchored with ^ and $")
        re.compile(value)
        return value

    @field_validator("supabase_url", mode="before")
    @classmethod
    def _normalise_url(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip().rstrip("/")
            return value or None
        return value

    @field_validator("llm_provider", "llm_fallback_provider", "embedding_provider", "auth_mode", "app_env", mode="before")
    @classmethod
    def _lowercase_enum(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip().lower()
            return value or None  # blank optional value means "not set"
        return value

    @field_validator("supabase_profiles_table")
    @classmethod
    def _safe_table_name(cls, value: str) -> str:
        # The table name is interpolated into a REST path; allow identifiers only.
        if not value.replace("_", "").isalnum():
            raise ValueError("SUPABASE_PROFILES_TABLE must be a plain identifier")
        return value

    @model_validator(mode="after")
    def _validate_combinations(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.llm_fallback_provider == self.llm_provider:
            raise ValueError("LLM_FALLBACK_PROVIDER must differ from LLM_PROVIDER (or be left blank)")
        if self.supabase_url and urlparse(self.supabase_url).path not in ("", "/"):
            raise ValueError("SUPABASE_URL must be the project base URL only (e.g. https://<ref>.supabase.co), without /rest/v1 or other paths")
        if self.supabase_url and not self.supabase_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("SUPABASE_URL must use https (http is only allowed for localhost)")
        if self.auth_mode == AuthMode.DEMO:
            if not self.allow_demo_auth:
                raise ValueError("AUTH_MODE=demo requires ALLOW_DEMO_AUTH=true")
            if self.app_env == AppEnv.PRODUCTION:
                raise ValueError(
                    "Demo authentication is forbidden when APP_ENV=production. "
                    "Set APP_ENV=development explicitly for local demos."
                )
        return self

    # --- Derived paths ---------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        path = Path(self.data_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def demo_auth_enabled(self) -> bool:
        return (
            self.auth_mode == AuthMode.DEMO
            and self.allow_demo_auth
            and self.app_env != AppEnv.PRODUCTION
        )

    # --- Lazy secret checks ----------------------------------------------------
    def require_gemini_key(self) -> str:
        if self.gemini_api_key is None:
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. It is required for embeddings "
                "(ingestion and querying) and for LLM_PROVIDER=gemini."
            )
        return self.gemini_api_key.get_secret_value()

    def require_groq_key(self) -> str:
        if self.groq_api_key is None:
            raise ConfigurationError("GROQ_API_KEY is not set but LLM_PROVIDER=groq.")
        return self.groq_api_key.get_secret_value()

    def require_supabase_service(self) -> tuple[str, str]:
        """Return (url, service_role_key) for backend-only access (vector store)."""
        missing = [
            name
            for name, value in (
                ("SUPABASE_URL", self.supabase_url),
                ("SUPABASE_SERVICE_ROLE_KEY", self.supabase_service_role_key),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(f"Supabase is not configured; missing: {', '.join(missing)}")
        assert self.supabase_url and self.supabase_service_role_key
        return self.supabase_url, self.supabase_service_role_key.get_secret_value()

    def require_supabase(self) -> tuple[str, str, str]:
        """Return (url, anon_key, service_role_key) or raise."""
        missing = [
            name
            for name, value in (
                ("SUPABASE_URL", self.supabase_url),
                ("SUPABASE_ANON_KEY", self.supabase_anon_key),
                ("SUPABASE_SERVICE_ROLE_KEY", self.supabase_service_role_key),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(f"Supabase is not configured; missing: {', '.join(missing)}")
        assert self.supabase_url and self.supabase_anon_key and self.supabase_service_role_key
        return (
            self.supabase_url,
            self.supabase_anon_key.get_secret_value(),
            self.supabase_service_role_key.get_secret_value(),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance (call ``get_settings.cache_clear()`` in tests)."""
    return Settings()

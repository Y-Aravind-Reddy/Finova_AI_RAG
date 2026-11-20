"""Shared fixtures. No test makes network calls or needs API keys."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Never let a developer's real .env/OS settings leak into tests.
os.environ["APP_ENV"] = "test"
os.environ["AUTH_MODE"] = "supabase"
os.environ["ALLOW_DEMO_AUTH"] = "false"

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import Field  # noqa: E402

from src.config import Settings  # noqa: E402

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


@pytest.fixture
def test_settings(tmp_path: Path, data_copy: Path) -> Settings:
    return make_settings(
        vectorstore_path=str(tmp_path / "vectorstore"),
        data_path=str(data_copy),
        chunk_size=500,
        chunk_overlap=100,
        retrieval_top_k=4,
    )


@pytest.fixture
def embeddings() -> KeywordEmbeddings:
    return KeywordEmbeddings()

"""Gemini embedding initialisation.

The same embedding model must be used to build and to query an index. The model
name is written to each index manifest at ingestion time and checked again at
query time (see ``rag_pipeline``), so a silently changed GEMINI_EMBEDDING_MODEL
fails loudly instead of returning garbage results.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from src.config import ConfigurationError, EmbeddingProvider, Settings, get_settings


def embedding_model_id(settings: Settings | None = None) -> str:
    """Stable identifier of the configured embedding model (stored in manifests)."""
    settings = settings or get_settings()
    name = settings.gemini_embedding_model
    if not name.startswith("models/"):
        name = f"models/{name}"
    return f"{settings.embedding_provider.value}:{name}"


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Create the configured embeddings client.

    Raises:
        ConfigurationError: if the provider is unsupported or the API key is missing.
    """
    settings = settings or get_settings()
    if settings.embedding_provider != EmbeddingProvider.GEMINI:
        raise ConfigurationError(
            f"Unsupported EMBEDDING_PROVIDER '{settings.embedding_provider}'. Only 'gemini' is supported."
        )
    api_key = settings.require_gemini_key()

    # Imported lazily so the API can start (e.g. /health) without the SDK loaded.
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    model = settings.gemini_embedding_model
    if not model.startswith("models/"):
        model = f"models/{model}"
    return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

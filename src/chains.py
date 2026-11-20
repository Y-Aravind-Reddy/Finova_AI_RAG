"""Gemini/Groq LLM selection and RAG chain construction."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from src.config import ConfigurationError, LLMProvider, Settings, get_settings
from src.prompts import build_rag_prompt


def get_llm(settings: Settings | None = None, provider: LLMProvider | None = None) -> BaseChatModel:
    """Instantiate the chat model for the configured (or given) provider.

    Raises:
        ConfigurationError: if the provider's API key is missing or the provider is unknown.
    """
    settings = settings or get_settings()
    provider = provider or settings.llm_provider

    if provider == LLMProvider.GEMINI:
        api_key = settings.require_gemini_key()
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=api_key,
            temperature=settings.llm_temperature,
        )

    if provider == LLMProvider.GROQ:
        api_key = settings.require_groq_key()
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_model,
            api_key=api_key,
            temperature=settings.llm_temperature,
        )

    raise ConfigurationError(f"Unsupported LLM_PROVIDER '{provider}'.")


def build_rag_chain(llm: BaseChatModel) -> Runnable:
    """prompt -> LLM -> plain string. Inputs: see ``prompts.prompt_variables``."""
    return build_rag_prompt() | llm | StrOutputParser()

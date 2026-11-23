"""Pick the answer-LLM provider from settings."""
from typing import Any

from app.clients.cohere_client import LLMClient
from app.clients.llm_openai import OpenAICompatLLM
from app.config import Settings

OPENAI_COMPATIBLE = {"openai_compatible", "openai-compatible", "openai"}


def make_llm(settings: Settings) -> Any:
    """Return the configured LLM client.

    LLM_PROVIDER=cohere (default) uses Cohere's chat API with the Cohere key.
    LLM_PROVIDER=openai_compatible uses LLM_BASE_URL + LLM_API_KEY + LLM_MODEL, which works with
    Groq, Google Gemini (OpenAI-compatible endpoint), OpenRouter, OpenAI and similar services.
    """
    provider = settings.llm_provider.strip().lower()
    if provider == "cohere":
        return LLMClient(settings)
    if provider in OPENAI_COMPATIBLE:
        if not settings.llm_base_url or not settings.llm_api_key.get_secret_value():
            raise RuntimeError(
                "LLM_PROVIDER=openai_compatible needs LLM_BASE_URL and LLM_API_KEY (and LLM_MODEL) in .env."
            )
        return OpenAICompatLLM(settings)
    raise RuntimeError(f"Unknown LLM_PROVIDER {settings.llm_provider!r}; use 'cohere' or 'openai_compatible'.")

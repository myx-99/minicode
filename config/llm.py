"""LLM factory — create a LangChain ChatModel from Settings."""

import os
from langchain_core.language_models import BaseChatModel
from config.settings import settings


def create_llm() -> BaseChatModel:
    """Create a LangChain ChatModel based on current Settings.

    Returns:
        A BaseChatModel instance ready for tool binding and invocation.

    Raises:
        ValueError: If the configured provider is unsupported.
        RuntimeError: If required API keys are missing.
    """
    provider = settings.llm_provider

    if provider == "openai":
        return _create_openai_llm()
    elif provider == "anthropic":
        return _create_anthropic_llm()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _create_openai_llm() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it in .env or as an environment variable."
        )

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=api_key,
        base_url=settings.openai_api_base,
        temperature=0.0,  # deterministic for coding tasks
    )


def _create_anthropic_llm() -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    api_key = settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. "
            "Set it in .env or as an environment variable."
        )

    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=api_key,
        temperature=0.0,
    )

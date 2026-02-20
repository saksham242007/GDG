"""
Centralized configuration for the Compliance Monitoring System.

All settings are read from environment variables (or .env file via python-dotenv).
This module provides:
  - LLM provider selection (OpenAI or Anthropic/Claude)
  - Database URL configuration (SQLite, PostgreSQL, MySQL)
  - Embedding model selection
  - ChromaDB directory path
  - A shared `get_llm()` factory and `get_embeddings()` factory
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

# Auto-load .env file (if present) before reading any env vars
from dotenv import load_dotenv

load_dotenv()

from src.utils.logging_config import get_logger


logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# Optional: custom base URL for OpenAI-compatible APIs (e.g. OpenRouter)
OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL", None)

# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# ---------------------------------------------------------------------------
# Database Configuration
# ---------------------------------------------------------------------------
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{PROJECT_ROOT / 'database' / 'compliance_rules.db'}",
)

# ---------------------------------------------------------------------------
# ChromaDB Configuration
# ---------------------------------------------------------------------------
CHROMA_DIR: Path = Path(
    os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma_db"))
)

# ---------------------------------------------------------------------------
# Confidence Threshold for Human Review Routing
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))


# ---------------------------------------------------------------------------
# Factory: get_llm()
# ---------------------------------------------------------------------------
def get_llm():
    """
    Return a configured LangChain Chat model based on LLM_PROVIDER.

    Supported providers:
      - "openai"    → ChatOpenAI  (requires OPENAI_API_KEY)
      - "anthropic" → ChatAnthropic (requires ANTHROPIC_API_KEY)

    Returns:
        A LangChain BaseChatModel instance.
    """
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        logger.info("Using OpenAI LLM: %s (base_url=%s)", LLM_MODEL, OPENAI_BASE_URL or "default")
        kwargs = dict(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        if OPENAI_BASE_URL:
            kwargs["openai_api_base"] = OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)

    elif LLM_PROVIDER == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is not installed. "
                "Run: pip install langchain-anthropic"
            )

        logger.info("Using Anthropic LLM: %s", LLM_MODEL)
        return ChatAnthropic(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: '{LLM_PROVIDER}'. "
            f"Supported: 'openai', 'anthropic'"
        )


# ---------------------------------------------------------------------------
# Factory: get_embeddings()
# ---------------------------------------------------------------------------
def get_embeddings():
    """
    Return a configured LangChain Embeddings model.

    Currently supports OpenAI embeddings only. Can be extended for
    other providers (Cohere, HuggingFace, etc.).

    Returns:
        A LangChain Embeddings instance.
    """
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        logger.info("Using OpenAI Embeddings: %s", EMBEDDING_MODEL)
        kwargs = dict(model=EMBEDDING_MODEL)
        if OPENAI_BASE_URL:
            kwargs["openai_api_base"] = OPENAI_BASE_URL
        return OpenAIEmbeddings(**kwargs)
    else:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER: '{EMBEDDING_PROVIDER}'. "
            f"Supported: 'openai'"
        )

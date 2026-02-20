"""
Unit tests for the Config module.

Tests the LLM factory, embeddings factory, and env-based configuration.
"""

import os
import pytest
from unittest.mock import patch


class TestConfigDefaults:
    """Test that default config values are set correctly."""

    def test_default_llm_provider(self):
        from src.config import LLM_PROVIDER
        assert LLM_PROVIDER in ("openai", "anthropic")

    def test_default_llm_model(self):
        from src.config import LLM_MODEL
        assert isinstance(LLM_MODEL, str)
        assert len(LLM_MODEL) > 0

    def test_default_database_url(self):
        from src.config import DATABASE_URL
        assert isinstance(DATABASE_URL, str)
        assert "compliance_rules" in DATABASE_URL or "://" in DATABASE_URL

    def test_default_confidence_threshold(self):
        from src.config import CONFIDENCE_THRESHOLD
        assert 0 <= CONFIDENCE_THRESHOLD <= 1


class TestGetLlm:
    """Test the get_llm() factory function."""

    @patch.dict(os.environ, {"LLM_PROVIDER": "invalid_provider"})
    def test_invalid_provider_raises(self):
        """Unknown LLM_PROVIDER should raise ValueError."""
        # Need to reimport to pick up env changes
        import importlib
        import src.config as config_mod
        importlib.reload(config_mod)
        with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
            config_mod.get_llm()


class TestGetEmbeddings:
    """Test the get_embeddings() factory function."""

    @patch.dict(os.environ, {"EMBEDDING_PROVIDER": "invalid_provider"})
    def test_invalid_embedding_provider_raises(self):
        """Unknown EMBEDDING_PROVIDER should raise ValueError."""
        import importlib
        import src.config as config_mod
        importlib.reload(config_mod)
        with pytest.raises(ValueError, match="Unsupported EMBEDDING_PROVIDER"):
            config_mod.get_embeddings()

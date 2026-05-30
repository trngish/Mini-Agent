"""Tests for config environment variable security."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from mini_agent.config import (
    AgentConfig,
    CLIOverrideConfig,
    Config,
    LLMConfig,
    ToolsConfig,
)


class TestEnvVarSecurity:
    """Test environment variable handling security."""

    def test_production_api_key_env_override(self):
        """Test that API key from env takes precedence (expected behavior).

        Environment variables override YAML file values when loading via from_yaml().
        Direct Config instantiation does NOT apply env overrides.
        """
        import tempfile
        from unittest.mock import patch
        import yaml

        # Create a temporary YAML file with a file-based API key
        yaml_content = """
api_key: "file-key-12345678"
api_base: "https://api.test.com"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(yaml_content)
            temp_path = f.name

        # Set env var before loading
        os.environ["MINIMAX_API_KEY"] = "env-api-key-12345678"
        try:
            # When loading from YAML, env var should override file value
            with patch("pathlib.Path.exists", return_value=True):
                config = Config.from_yaml(temp_path)
            assert config.llm.api_key == "env-api-key-12345678"
        finally:
            del os.environ["MINIMAX_API_KEY"]
            Path(temp_path).unlink(missing_ok=True)

    def test_env_override_validation(self):
        """Test that CLI override values are re-validated."""
        config = Config(
            llm=LLMConfig(api_key="sk-test-key-12345678", api_base="https://api.test.com"),
            agent=AgentConfig(),
            tools=ToolsConfig(),
        )

        # CLI override should trigger re-validation
        cli_override = CLIOverrideConfig(max_steps=200)
        config.merge_cli_overrides(cli_override)
        assert config.agent.max_steps == 200

    def test_invalid_api_key_length_rejected(self):
        """Test that short API keys are rejected."""
        with pytest.raises(ValidationError, match=r"至少8个字符|at least 8 characters"):
            LLMConfig(api_key="short", api_base="https://api.test.com")
"""Tests for config validator utilities."""

from mini_agent.utils.config_validator import (
    ConfigValidationError,
    ConfigValidator,
)


class TestSanitizeTimeout:
    """Tests for ConfigValidator.sanitize_timeout method."""

    def test_valid_timeout(self):
        assert ConfigValidator.sanitize_timeout(120) == 120

    def test_too_small_timeout(self):
        assert ConfigValidator.sanitize_timeout(0) == 120  # Returns default

    def test_too_large_timeout(self):
        assert ConfigValidator.sanitize_timeout(9999) == 120  # Returns default

    def test_custom_default(self):
        assert ConfigValidator.sanitize_timeout(0, default=60) == 60

    def test_custom_range(self):
        assert ConfigValidator.sanitize_timeout(50, min_val=10, max_val=100) == 50

    def test_outside_custom_range(self):
        assert ConfigValidator.sanitize_timeout(5, min_val=10, max_val=100) == 120


class TestSanitizeTokenCount:
    """Tests for ConfigValidator.sanitize_token_count method."""

    def test_valid_count(self):
        assert ConfigValidator.sanitize_token_count(500_000) == 500_000

    def test_too_small_count(self):
        assert ConfigValidator.sanitize_token_count(100) == 800_000  # Returns default

    def test_too_large_count(self):
        assert ConfigValidator.sanitize_token_count(2_000_000) == 800_000  # Returns default


class TestClamp:
    """Tests for ConfigValidator.clamp method."""

    def test_within_range(self):
        assert ConfigValidator.clamp(5.0, 0.0, 10.0) == 5.0

    def test_below_min(self):
        assert ConfigValidator.clamp(-1.0, 0.0, 10.0) == 0.0

    def test_above_max(self):
        assert ConfigValidator.clamp(15.0, 0.0, 10.0) == 10.0

    def test_exact_min(self):
        assert ConfigValidator.clamp(0.0, 0.0, 10.0) == 0.0

    def test_exact_max(self):
        assert ConfigValidator.clamp(10.0, 0.0, 10.0) == 10.0


class TestConfigValidator:
    """Tests for ConfigValidator class."""

    def test_validate_empty_config(self):
        """ConfigValidator should handle missing fields gracefully."""
        from unittest.mock import MagicMock

        config = MagicMock()
        # Make _get_nested_value return None for all paths
        config.tools = None
        config.m27 = None
        config.llm = None
        errors = ConfigValidator.validate(config)
        # Missing fields should be skipped, not error
        assert isinstance(errors, list)

    def test_validate_or_raise_no_errors(self):
        from unittest.mock import MagicMock

        config = MagicMock()
        config.tools = None
        config.m27 = None
        config.llm = None
        # Should not raise for None fields
        ConfigValidator.validate_or_raise(config)

    def test_max_concurrent_tools_range(self):
        """Verify that max_concurrent_tools range allows the default value of 20."""
        rule = [r for r in ConfigValidator.RULES if r.field == "m27.max_concurrent_tools"][0]
        assert rule.validate(20) is True, "Default value of 20 should be within valid range"
        assert rule.validate(1) is True
        assert rule.validate(30) is True

    def test_thinking_budget_range(self):
        rule = [r for r in ConfigValidator.RULES if r.field == "m27.thinking_budget_tokens"][0]
        assert rule.validate(0) is True
        assert rule.validate(32768) is True

    def test_max_steps_range(self):
        rule = [r for r in ConfigValidator.RULES if r.field == "agent.max_steps"][0]
        assert rule.validate(1) is True
        assert rule.validate(1000) is True
        assert rule.validate(0) is False


class TestConfigValidationError:
    """Tests for ConfigValidationError."""

    def test_error_fields(self):
        error = ConfigValidationError(field="test", message="test error", value=42)
        assert error.field == "test"
        assert error.message == "test error"
        assert error.value == 42

    def test_error_string(self):
        error = ConfigValidationError(field="test", message="test error")
        assert "test" in str(error)
        assert "test error" in str(error)

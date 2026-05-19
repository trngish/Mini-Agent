"""Configuration validation module.

Provides validation for configuration values with range checks and constraints.
"""

from dataclasses import dataclass
from typing import Any


class ConfigValidationError(Exception):
    """Configuration validation error."""
    
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation error for '{field}': {message}")


@dataclass
class ValidationRule:
    """A single validation rule."""
    field: str
    validate: callable
    message: str


class ConfigValidator:
    """Configuration validator with range checks and constraints.
    
    Provides validation for timeout values, token limits, and other
    configuration parameters to ensure they are within acceptable ranges.
    """
    
    # Validation rules
    RULES: list[ValidationRule] = [
        ValidationRule(
            field="tools.bash_timeout",
            validate=lambda v: 1 <= v <= 600,
            message="Must be between 1 and 600 seconds",
        ),
        ValidationRule(
            field="tools.mcp.connect_timeout",
            validate=lambda v: 1 <= v <= 60,
            message="Must be between 1 and 60 seconds",
        ),
        ValidationRule(
            field="tools.mcp.execute_timeout",
            validate=lambda v: 1 <= v <= 300,
            message="Must be between 1 and 300 seconds",
        ),
        ValidationRule(
            field="tools.mcp.sse_read_timeout",
            validate=lambda v: 1 <= v <= 600,
            message="Must be between 1 and 600 seconds",
        ),
        ValidationRule(
            field="agent.max_steps",
            validate=lambda v: 1 <= v <= 1000,
            message="Must be between 1 and 3000",
        ),
        ValidationRule(
            field="m27.thinking_budget_tokens",
            validate=lambda v: 0 <= v <= 32768,
            message="Must be between 0 and 32768 tokens",
        ),
        ValidationRule(
            field="m27.max_output_tokens",
            validate=lambda v: 1024 <= v <= 32768,
            message="Must be between 1024 and 32768 tokens",
        ),
        ValidationRule(
            field="m27.token_limit",
            validate=lambda v: 100_000 <= v <= 1_000_000,
            message="Must be between 100K and 1M tokens",
        ),
        ValidationRule(
            field="m27.max_concurrent_tools",
            validate=lambda v: 1 <= v <= 30,
            message="Must be between 1 and 30",
        ),
        ValidationRule(
            field="llm.retry.max_retries",
            validate=lambda v: 0 <= v <= 10,
            message="Must be between 0 and 10",
        ),
        ValidationRule(
            field="llm.retry.initial_delay",
            validate=lambda v: 0.1 <= v <= 60,
            message="Must be between 0.1 and 60 seconds",
        ),
        ValidationRule(
            field="llm.retry.max_delay",
            validate=lambda v: 1 <= v <= 300,
            message="Must be between 1 and 300 seconds",
        ),
    ]
    
    @classmethod
    def validate(cls, config: Any) -> list[ConfigValidationError]:
        """Validate configuration object.
        
        Args:
            config: Config object with validation rules applied
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        for rule in cls.RULES:
            try:
                value = cls._get_nested_value(config, rule.field)
                if value is not None and not rule.validate(value):
                    errors.append(ConfigValidationError(
                        field=rule.field,
                        message=rule.message,
                        value=value,
                    ))
            except (AttributeError, KeyError, TypeError):
                # Skip fields that don't exist
                pass
        
        return errors
    
    @classmethod
    def _get_nested_value(cls, obj: Any, path: str) -> Any:
        """Get nested value from object using dot notation.
        
        Args:
            obj: Object to traverse
            path: Dot-separated path (e.g., "tools.bash_timeout")
            
        Returns:
            Value at path, or None if not found
        """
        parts = path.split(".")
        current = obj
        
        for part in parts:
            if current is None:
                return None
            current = getattr(current, part, None)
        
        return current
    
    @classmethod
    def validate_or_raise(cls, config: Any) -> None:
        """Validate configuration or raise on error.
        
        Args:
            config: Config object to validate
            
        Raises:
            ConfigValidationError: If any validation fails
        """
        errors = cls.validate(config)
        if errors:
            messages = "\n".join(f"  - {e.field}: {e.message}" for e in errors)
            raise ConfigValidationError(
                field="multiple",
                message=f"Configuration validation failed:\n{messages}",
            )
    
    @classmethod
    def sanitize_timeout(cls, timeout: int, default: int = 120, min_val: int = 1, max_val: int = 600) -> int:
        """Sanitize timeout value to be within acceptable range.
        
        Args:
            timeout: Raw timeout value
            default: Default if invalid
            min_val: Minimum acceptable value
            max_val: Maximum acceptable value
            
        Returns:
            Sanitized timeout
        """
        if timeout < min_val or timeout > max_val:
            return default
        return timeout
    
    @classmethod
    def sanitize_token_count(cls, tokens: int, default: int = 800_000, min_val: int = 1000, max_val: int = 1_000_000) -> int:
        """Sanitize token count to be within acceptable range.
        
        Args:
            tokens: Raw token count
            default: Default if invalid
            min_val: Minimum acceptable value
            max_val: Maximum acceptable value
            
        Returns:
            Sanitized token count
        """
        if tokens < min_val or tokens > max_val:
            return default
        return tokens
    
    @classmethod
    def clamp(cls, value: float, min_val: float, max_val: float) -> float:
        """Clamp value to range.
        
        Args:
            value: Value to clamp
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Clamped value
        """
        return max(min_val, min(max_val, value))
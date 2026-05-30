"""配置验证模块。

提供配置值的验证，包含范围检查和约束条件。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ConfigValidationError(Exception):
    """配置验证错误。"""

    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation error for '{field}': {message}")


@dataclass
class ValidationRule:
    """单个验证规则。"""

    field: str
    validate: Callable[[Any], bool]
    message: str


class ConfigValidator:
    """配置验证器，包含范围检查和约束条件。

    提供超时值、令牌限制和其他配置参数的验证，
    确保它们在可接受的范围内。
    """

    # 验证规则
    RULES: list[ValidationRule] = [
        ValidationRule(
            field="tools.bash_timeout",
            validate=lambda v: 1 <= v <= 600,
            message="必须在 1 到 600 秒之间",
        ),
        ValidationRule(
            field="tools.mcp.connect_timeout",
            validate=lambda v: 1 <= v <= 60,
            message="必须在 1 到 60 秒之间",
        ),
        ValidationRule(
            field="tools.mcp.execute_timeout",
            validate=lambda v: 1 <= v <= 300,
            message="必须在 1 到 300 秒之间",
        ),
        ValidationRule(
            field="tools.mcp.sse_read_timeout",
            validate=lambda v: 1 <= v <= 600,
            message="必须在 1 到 600 秒之间",
        ),
        ValidationRule(
            field="agent.max_steps",
            validate=lambda v: 1 <= v <= 1000,
            message="必须在 1 到 3000 之间",
        ),
        ValidationRule(
            field="m27.thinking_budget_tokens",
            validate=lambda v: 0 <= v <= 32768,
            message="必须在 0 到 32768 tokens 之间",
        ),
        ValidationRule(
            field="m27.max_output_tokens",
            validate=lambda v: 1024 <= v <= 32768,
            message="必须在 1024 到 32768 tokens 之间",
        ),
        ValidationRule(
            field="m27.token_limit",
            validate=lambda v: 100_000 <= v <= 1_000_000,
            message="必须在 100K 到 1M tokens 之间",
        ),
        ValidationRule(
            field="m27.max_concurrent_tools",
            validate=lambda v: 1 <= v <= 30,
            message="必须在 1 到 30 之间",
        ),
        ValidationRule(
            field="llm.retry.max_retries",
            validate=lambda v: 0 <= v <= 10,
            message="必须在 0 到 10 之间",
        ),
        ValidationRule(
            field="llm.retry.initial_delay",
            validate=lambda v: 0.1 <= v <= 60,
            message="必须在 0.1 到 60 秒之间",
        ),
        ValidationRule(
            field="llm.retry.max_delay",
            validate=lambda v: 1 <= v <= 300,
            message="必须在 1 到 300 秒之间",
        ),
    ]

    @classmethod
    def validate(cls, config: Any) -> list[ConfigValidationError]:
        """验证配置对象。

        Args:
            config: 应用了验证规则的配置对象

        Returns:
            验证错误列表（如果有效则为空）
        """
        errors = []

        for rule in cls.RULES:
            try:
                value = cls._get_nested_value(config, rule.field)
                if value is not None and not rule.validate(value):
                    errors.append(
                        ConfigValidationError(
                            field=rule.field,
                            message=rule.message,
                            value=value,
                        )
                    )
            except (AttributeError, KeyError, TypeError):
                # 跳过不存在的字段
                pass

        return errors

    @classmethod
    def _get_nested_value(cls, obj: Any, path: str) -> Any:
        """使用点号表示法从对象获取嵌套值。

        Args:
            obj: 要遍历的对象
            path: 点分隔的路径（例如 "tools.bash_timeout"）

        Returns:
            路径处的值，如果未找到则返回 None
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
        """验证配置，验证失败则抛出异常。

        Args:
            config: 要验证的配置对象

        Raises:
            ConfigValidationError: 如果任何验证失败
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
        """清理超时值以使其在可接受范围内。

        Args:
            timeout: 原始超时值
            default: 无效时的默认值
            min_val: 最小可接受值
            max_val: 最大可接受值

        Returns:
            清理后的超时值
        """
        if timeout < min_val or timeout > max_val:
            return default
        return timeout

    @classmethod
    def sanitize_token_count(
        cls, tokens: int, default: int = 800_000, min_val: int = 1000, max_val: int = 1_000_000
    ) -> int:
        """清理令牌计数以使其在可接受范围内。

        Args:
            tokens: 原始令牌计数
            default: 无效时的默认值
            min_val: 最小可接受值
            max_val: 最大可接受值

        Returns:
            清理后的令牌计数
        """
        if tokens < min_val or tokens > max_val:
            return default
        return tokens

    @classmethod
    def clamp(cls, value: float, min_val: float, max_val: float) -> float:
        """将值限制在指定范围内。

        Args:
            value: 要限制的值
            min_val: 最小值
            max_val: 最大值

        Returns:
            限制后的值
        """
        return max(min_val, min(max_val, value))

"""Utility modules for Mini-Agent."""

from .display import (
    Colors,
    BoxDrawing,
    colorize,
    create_progress_bar,
    format_table_row,
    create_divider,
)
from .terminal_utils import (
    calculate_display_width,
    pad_to_width,
    truncate_with_ellipsis,
)
from .token_utils import (
    get_encoder,
    encode_text,
    count_tokens,
    decode_tokens,
    get_tokens_info,
    clear_encoder_cache,
    get_cache_size,
    preload_encoders,
    get_cl100k_base,
)
from .logging_config import get_logger, AgentLoggingConfig
from .message_validator import MessageValidator, ValidationError
from .error_handler import (
    LLMErrorClassifier,
    LLMError,
    LLMErrorType,
    format_llm_error,
)
from .m27_optimization import (
    M27PromptOptimizer,
    M27ContextManager,
    M27ToolOptimizer,
    M27AgentTeams,
)
from .model_utils import (
    is_m27_model,
    is_minimax_model,
    get_model_specs,
    get_thinking_budget,
    get_max_output_tokens,
    get_token_limit_for_model,
    ModelSpecs,
    # Legacy alias for backwards compatibility
    is_m27_enabled,
)
from .platform_utils import (
    PlatformUtils,
    PlatformInfo,
    get_platform_shell_args,
    get_subprocess_env,
    normalize_path_separators,
)
from .config_validator import ConfigValidator, ConfigValidationError

__all__ = [
    # Display utilities
    "Colors",
    "BoxDrawing",
    "colorize",
    "create_progress_bar",
    "format_table_row",
    "create_divider",
    # Terminal utilities
    "calculate_display_width",
    "pad_to_width",
    "truncate_with_ellipsis",
    # Token utilities
    "get_encoder",
    "encode_text",
    "count_tokens",
    "decode_tokens",
    "get_tokens_info",
    "clear_encoder_cache",
    "get_cache_size",
    "preload_encoders",
    "get_cl100k_base",
    # Logging
    "get_logger",
    "AgentLoggingConfig",
    # Message validation
    "MessageValidator",
    "ValidationError",
    # Error handling
    "LLMErrorClassifier",
    "LLMError",
    "LLMErrorType",
    "format_llm_error",
    # M2.7 optimizations
    "M27PromptOptimizer",
    "M27ContextManager",
    "M27ToolOptimizer",
    "M27AgentTeams",
    # Model utilities (single source of truth)
    "is_m27_model",
    "is_minimax_model",
    "get_model_specs",
    "get_thinking_budget",
    "get_max_output_tokens",
    "get_token_limit_for_model",
    "is_m27_enabled",  # Legacy alias
    "ModelSpecs",
    # Platform utilities (single source of truth)
    "PlatformUtils",
    "PlatformInfo",
    "get_platform_shell_args",
    "get_subprocess_env",
    "normalize_path_separators",
    # Config validation
    "ConfigValidator",
    "ConfigValidationError",
]
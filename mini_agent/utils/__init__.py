"""Mini-Agent 的工具模块。"""

from .config_validator import ConfigValidationError, ConfigValidator
from .context_cache import ContextCache, get_cache_hit_rate, get_context_cache, record_cache_access
from .display import (
    BoxDrawing,
    Colors,
    colorize,
    create_divider,
    create_progress_bar,
    format_table_row,
)
from .error_handler import (
    LLMError,
    LLMErrorClassifier,
    LLMErrorType,
    format_llm_error,
)
from .logging_config import AgentLoggingConfig, get_logger
from .m27_optimization import (
    M27AgentTeams,
    M27ContextManager,
    M27PromptOptimizer,
    M27ToolOptimizer,
)
from .message_validator import MessageValidator, ValidationError
from .model_utils import (
    ModelSpecs,
    get_max_output_tokens,
    get_model_specs,
    get_thinking_budget,
    get_token_limit_for_model,
    # Legacy alias for backwards compatibility
    is_m27_enabled,
    is_m27_model,
    is_minimax_model,
)
from .platform_utils import (
    PlatformInfo,
    PlatformUtils,
    get_platform_shell_args,
    get_subprocess_env,
    normalize_path_separators,
)
from .summary_manager import AdaptiveSummaryManager, MessageComplexityAnalyzer
from .terminal_utils import (
    calculate_display_width,
    pad_to_width,
    truncate_with_ellipsis,
)
from .thinking_manager import ThinkingManager
from .token_utils import (
    clear_encoder_cache,
    count_tokens,
    decode_tokens,
    encode_text,
    get_cache_size,
    get_cl100k_base,
    get_encoder,
    get_tokens_info,
    preload_encoders,
)
from .tool_group_optimizer import ToolGroupOptimizer

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
    # Context management (new optimizations)
    "ContextCache",
    "get_context_cache",
    "record_cache_access",
    "get_cache_hit_rate",
    "ToolGroupOptimizer",
    "AdaptiveSummaryManager",
    "MessageComplexityAnalyzer",
    "ThinkingManager",
]

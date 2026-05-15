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
    M27Config,
    M27PromptOptimizer,
    M27ContextManager,
    M27ToolOptimizer,
    is_m27_enabled,
)

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
    "M27Config",
    "M27PromptOptimizer",
    "M27ContextManager",
    "M27ToolOptimizer",
    "is_m27_enabled",
]


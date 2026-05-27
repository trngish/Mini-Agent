"""Model utilities for unified model detection and configuration.

Provides single source of truth for model type detection and optimization settings.
"""

from dataclasses import dataclass

# Model identifiers for M2.7
M27_MODEL_IDENTIFIERS = (
    "MiniMax-M2.7",
    "MiniMax-M2",
    "MiniMax-M2.5",
)

# Model identifiers for other MiniMax models
MINIMAX_MODEL_IDENTIFIERS = (
    "MiniMax",
    "MiniMax-M2",
    "MiniMax-M2.5",
    "MiniMax-M2.7",
)


@dataclass(frozen=True)
class ModelSpecs:
    """Model specifications container.

    Contains token limits and capabilities for a specific model.
    """

    max_output_tokens: int
    max_context_tokens: int
    supports_extended_thinking: bool
    max_thinking_budget: int
    supports_parallel_tools: bool
    default_thinking_budget: int


# Default specs per model family
DEFAULT_MODEL_SPECS: dict[str, ModelSpecs] = {
    # M2.7 specifications
    "MiniMax-M2.7": ModelSpecs(
        max_output_tokens=32768,
        max_context_tokens=1_000_000,
        supports_extended_thinking=True,
        max_thinking_budget=32768,
        supports_parallel_tools=True,
        default_thinking_budget=16384,
    ),
    # M2.5 specifications
    "MiniMax-M2.5": ModelSpecs(
        max_output_tokens=8192,
        max_context_tokens=1_000_000,
        supports_extended_thinking=True,
        max_thinking_budget=8192,
        supports_parallel_tools=True,
        default_thinking_budget=8192,
    ),
    # M2 specifications
    "MiniMax-M2": ModelSpecs(
        max_output_tokens=8192,
        max_context_tokens=128_000,
        supports_extended_thinking=True,
        max_thinking_budget=8192,
        supports_parallel_tools=True,
        default_thinking_budget=8192,
    ),
    # Fallback for unknown models
    "default": ModelSpecs(
        max_output_tokens=4096,
        max_context_tokens=128_000,
        supports_extended_thinking=False,
        max_thinking_budget=0,
        supports_parallel_tools=False,
        default_thinking_budget=0,
    ),
}


def is_m27_model(model_name: str) -> bool:
    """Check if the given model name is an M2.7 variant.

    Single source of truth for M2.7 detection.

    Args:
        model_name: The model name to check

    Returns:
        True if the model is an M2.7 variant
    """
    if not model_name:
        return False
    model_upper = model_name.upper()
    return any(identifier.upper() in model_upper for identifier in M27_MODEL_IDENTIFIERS)


def is_minimax_model(model_name: str) -> bool:
    """Check if the given model name is any MiniMax model.

    Args:
        model_name: The model name to check

    Returns:
        True if the model is a MiniMax variant
    """
    if not model_name:
        return False
    model_upper = model_name.upper()
    return any(identifier.upper() in model_upper for identifier in MINIMAX_MODEL_IDENTIFIERS)


def get_model_specs(model_name: str) -> ModelSpecs:
    """Get model specifications for the given model.

    Args:
        model_name: The model name

    Returns:
        ModelSpecs for the model, or default specs if unknown
    """
    if not model_name:
        return DEFAULT_MODEL_SPECS["default"]

    model_upper = model_name.upper()

    # Check exact matches first
    for key in DEFAULT_MODEL_SPECS:
        if key != "default" and key.upper() in model_upper:
            return DEFAULT_MODEL_SPECS[key]

    # Check partial matches
    if "M2.7" in model_upper:
        return DEFAULT_MODEL_SPECS["MiniMax-M2.7"]
    elif "M2.5" in model_upper:
        return DEFAULT_MODEL_SPECS["MiniMax-M2.5"]
    elif "M2" in model_upper:
        return DEFAULT_MODEL_SPECS["MiniMax-M2"]

    return DEFAULT_MODEL_SPECS["default"]


def is_extended_thinking_enabled(model_name: str, config_enabled: bool = True) -> bool:
    """Check if extended thinking should be enabled for the model.

    Args:
        model_name: The model name
        config_enabled: Configuration setting (if False, always returns False)

    Returns:
        True if extended thinking should be enabled
    """
    if not config_enabled:
        return False
    specs = get_model_specs(model_name)
    return specs.supports_extended_thinking


def get_thinking_budget(model_name: str, requested_budget: int) -> int:
    """Get constrained thinking budget for the model.

    Args:
        model_name: The model name
        requested_budget: The budget requested by configuration

    Returns:
        Constrained budget within model's max_thinking_budget
    """
    specs = get_model_specs(model_name)
    if not specs.supports_extended_thinking:
        return 0
    return min(requested_budget, specs.max_thinking_budget)


def get_max_output_tokens(model_name: str) -> int:
    """Get max output tokens for the model.

    Args:
        model_name: The model name

    Returns:
        Maximum output tokens
    """
    return get_model_specs(model_name).max_output_tokens


def get_token_limit_for_model(model_name: str, configured_limit: int | None = None) -> int:
    """Get token limit for context management.

    Args:
        model_name: The model name
        configured_limit: Optional configured limit override

    Returns:
        Token limit to use (80% of context window as safety margin)
    """
    specs = get_model_specs(model_name)

    if configured_limit is not None and configured_limit > 0:
        return configured_limit

    # Default to 80% of context window
    return int(specs.max_context_tokens * 0.8)


# Legacy alias for backwards compatibility
is_m27_enabled = is_m27_model

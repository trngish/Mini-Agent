"""模型工具，用于统一的模型检测和配置。

提供模型类型检测和优化设置的单一数据源。
"""

from dataclasses import dataclass

# M2.7 的模型标识符
M27_MODEL_IDENTIFIERS = (
    "MiniMax-M2.7",
    "MiniMax-M2",
    "MiniMax-M2.5",
)

# 其他 MiniMax 模型的模型标识符
MINIMAX_MODEL_IDENTIFIERS = (
    "MiniMax",
    "MiniMax-M2",
    "MiniMax-M2.5",
    "MiniMax-M2.7",
)


@dataclass(frozen=True)
class ModelSpecs:
    """模型规格容器。

    包含特定模型的令牌限制和能力。
    """

    max_output_tokens: int
    max_context_tokens: int
    supports_extended_thinking: bool
    max_thinking_budget: int
    supports_parallel_tools: bool
    default_thinking_budget: int


# 每个模型系列的默认规格
DEFAULT_MODEL_SPECS: dict[str, ModelSpecs] = {
    # M2.7 规格
    "MiniMax-M2.7": ModelSpecs(
        max_output_tokens=32768,
        max_context_tokens=1_000_000,
        supports_extended_thinking=True,
        max_thinking_budget=32768,
        supports_parallel_tools=True,
        default_thinking_budget=16384,
    ),
    # M2.5 规格
    "MiniMax-M2.5": ModelSpecs(
        max_output_tokens=8192,
        max_context_tokens=1_000_000,
        supports_extended_thinking=True,
        max_thinking_budget=8192,
        supports_parallel_tools=True,
        default_thinking_budget=8192,
    ),
    # M2 规格
    "MiniMax-M2": ModelSpecs(
        max_output_tokens=8192,
        max_context_tokens=128_000,
        supports_extended_thinking=True,
        max_thinking_budget=8192,
        supports_parallel_tools=True,
        default_thinking_budget=8192,
    ),
    # 未知模型的备用规格
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
    """检查给定的模型名称是否为 M2.7 变体。

    M2.7 检测的单一数据源。

    Args:
        model_name: 要检查的模型名称

    Returns:
        如果是 M2.7 变体则返回 True
    """
    if not model_name:
        return False
    model_upper = model_name.upper()
    return any(identifier.upper() in model_upper for identifier in M27_MODEL_IDENTIFIERS)


def is_minimax_model(model_name: str) -> bool:
    """检查给定的模型名称是否为任何 MiniMax 模型。

    Args:
        model_name: 要检查的模型名称

    Returns:
        如果是 MiniMax 变体则返回 True
    """
    if not model_name:
        return False
    model_upper = model_name.upper()
    return any(identifier.upper() in model_upper for identifier in MINIMAX_MODEL_IDENTIFIERS)


def get_model_specs(model_name: str) -> ModelSpecs:
    """获取给定模型的模型规格。

    Args:
        model_name: 模型名称

    Returns:
        模型的 ModelSpecs，如果是未知模型则返回默认规格
    """
    if not model_name:
        return DEFAULT_MODEL_SPECS["default"]

    model_upper = model_name.upper()

    # 首先检查精确匹配
    for key in DEFAULT_MODEL_SPECS:
        if key != "default" and key.upper() in model_upper:
            return DEFAULT_MODEL_SPECS[key]

    # 检查部分匹配
    if "M2.7" in model_upper:
        return DEFAULT_MODEL_SPECS["MiniMax-M2.7"]
    elif "M2.5" in model_upper:
        return DEFAULT_MODEL_SPECS["MiniMax-M2.5"]
    elif "M2" in model_upper:
        return DEFAULT_MODEL_SPECS["MiniMax-M2"]

    return DEFAULT_MODEL_SPECS["default"]


def is_extended_thinking_enabled(model_name: str, config_enabled: bool = True) -> bool:
    """检查是否为模型启用扩展思考。

    Args:
        model_name: 模型名称
        config_enabled: 配置设置（如果为 False，始终返回 False）

    Returns:
        如果应启用扩展思考则返回 True
    """
    if not config_enabled:
        return False
    specs = get_model_specs(model_name)
    return specs.supports_extended_thinking


def get_thinking_budget(model_name: str, requested_budget: int) -> int:
    """获取模型的约束思考预算。

    Args:
        model_name: 模型名称
        requested_budget: 配置请求的预算

    Returns:
        模型 max_thinking_budget 范围内的约束预算
    """
    specs = get_model_specs(model_name)
    if not specs.supports_extended_thinking:
        return 0
    return min(requested_budget, specs.max_thinking_budget)


def get_max_output_tokens(model_name: str) -> int:
    """获取模型的最大输出 token 数。

    Args:
        model_name: 模型名称

    Returns:
        最大输出 token 数
    """
    return get_model_specs(model_name).max_output_tokens


def get_token_limit_for_model(model_name: str, configured_limit: int | None = None) -> int:
    """获取用于上下文管理的 token 限制。

    Args:
        model_name: 模型名称
        configured_limit: 可选的配置限制覆盖

    Returns:
        要使用的 token 限制（上下文窗口的 80% 作为安全余量）
    """
    specs = get_model_specs(model_name)

    if configured_limit is not None and configured_limit > 0:
        return configured_limit

    # 默认为上下文窗口的 80%
    return int(specs.max_context_tokens * 0.8)


# 向下兼容的旧别名
is_m27_enabled = is_m27_model

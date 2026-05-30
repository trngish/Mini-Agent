"""M2.7 模型的思考预算管理

提供基于任务复杂度的自适应思考预算调整。
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_context import AgentContext

# 自适应思考预算级别（tokens）
# 更高的预算使复杂任务能够进行更深入的推理
THINKING_BUDGET_SIMPLE = 16384
THINKING_BUDGET_MEDIUM = 24576
THINKING_BUDGET_COMPLEX = 32768
THINKING_BUDGET_SUPER = 32768

# 用于自动检测的复杂度指标
COMPLEXITY_HIGH_KEYWORDS = {
    "refactor",
    "architecture",
    "rewrite",
    "migrate",
    "comprehensive",
    "entire",
    "all files",
    "batch",
    "multi-file",
    "design",
    "debug",
    "investigate",
    "optimize",
    "performance",
}

COMPLEXITY_MEDIUM_KEYWORDS = {
    "modify",
    "fix",
    "implement",
    "add",
    "update",
    "create",
    "search",
    "analyze",
    "check",
    "compare",
    "convert",
}


class ThinkingBudgetManager:
    """M2.7 模型的自适应思考预算管理器

    根据任务复杂度动态调整思考预算，
    以优化减少 API 调用次数（按次计费优化）。
    """

    def __init__(self, context: "AgentContext"):
        self._context = context
        self._max_budget: int = 16384
        self._current_budget: int = 16384

    def configure(self, max_budget: int, is_m27: bool) -> None:
        """使用智能体配置管理器

        参数:
            max_budget: 配置中的最大思考预算
            is_m27: 模型是否为 M2.7
        """
        self._max_budget = max_budget if is_m27 else 0
        self._current_budget = self._max_budget

    def adjust(self, user_message: str) -> None:
        """根据任务复杂度自适应调整思考预算

        按次计费优化：tokens 是免费的，
        更深入的思考 -> 更高的准确率 -> 更少的重试 -> 更少的总调用次数。
        即使是简单的任务也从足够的思考空间中受益，以提高单次调用完成率。

        参数:
            user_message: 要分析复杂度的用户消息
        """
        if not self._context.is_m27:
            return

        msg_lower = user_message.lower()

        # 从关键词检测复杂度级别
        high_matches = sum(1 for kw in COMPLEXITY_HIGH_KEYWORDS if kw in msg_lower)
        medium_matches = sum(1 for kw in COMPLEXITY_MEDIUM_KEYWORDS if kw in msg_lower)

        # 估算提到的文件数量
        file_mentions = len(
            re.findall(
                r"\.(py|js|ts|jsx|tsx|java|go|rs|c|cpp|h|rb|php|yaml|yml|json|toml|md|txt|csv|sql|sh|bash|ps1)",
                msg_lower,
            )
        )

        # 确定复杂度
        if high_matches >= 2 or file_mentions >= 4:
            new_budget = THINKING_BUDGET_SUPER
            level = "ultra-complex"
        elif high_matches >= 1 or file_mentions >= 2 or medium_matches >= 3:
            new_budget = THINKING_BUDGET_COMPLEX
            level = "complex"
        elif medium_matches >= 1 or file_mentions >= 1:
            new_budget = THINKING_BUDGET_MEDIUM
            level = "medium"
        else:
            new_budget = THINKING_BUDGET_SIMPLE
            level = "simple"

        # 同时将消息长度作为信号考虑
        msg_tokens = len(user_message) // 3  # 粗略估算
        if msg_tokens > 2000:
            new_budget = max(new_budget, THINKING_BUDGET_COMPLEX)
            level = "complex(long msg)"

        # 限制为配置中的最大预算
        new_budget = min(new_budget, self._max_budget)

        if new_budget != self._current_budget:
            old_budget = self._current_budget
            self._update_budget(new_budget)
            from ..utils.display import Colors

            print(f"{Colors.DIM}🧠 Thinking budget adjusted: {old_budget} → {new_budget} ({level} task){Colors.RESET}")

    def _update_budget(self, budget: int) -> None:
        """通过官方 API 更新思考预算

        参数:
            budget: 新的思考预算（以 tokens 为单位）
        """
        self._current_budget = budget
        # 同步到上下文，以便状态显示当前值
        self._context.thinking_budget = budget
        # 通过公共 API 更新 LLM 客户端的思考预算
        if self._context.llm and hasattr(self._context.llm, "configure_thinking_budget"):
            self._context.llm.configure_thinking_budget(budget)

    @property
    def current_budget(self) -> int:
        """获取当前思考预算"""
        return self._current_budget

    @property
    def max_budget(self) -> int:
        """获取最大思考预算"""
        return self._max_budget
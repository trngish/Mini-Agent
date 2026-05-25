"""Thinking budget management for M2.7 models.

Provides adaptive thinking budget adjustment based on task complexity.
按次数计费优化：token免费，思考越深→命中率越高→重试越少→总调用次数越少。
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import Agent

# Adaptive thinking budget levels (按次数计费优化：token免费，思考越深命中率越高)
# 提高基础预算：更深思考 → 更高命中率 → 更少重试 → 更少总调用次数
THINKING_BUDGET_SIMPLE = 16384      # 简单任务：原8K→16K，确保一次做对
THINKING_BUDGET_MEDIUM = 24576      # 中等任务：原16K→24K，减少返工
THINKING_BUDGET_COMPLEX = 32768     # 复杂任务：原24K→32K，深度规划
THINKING_BUDGET_SUPER = 32768       # 超复杂任务：32K上限

# Complexity indicators for auto-detection
COMPLEXITY_HIGH_KEYWORDS = {
    "重构", "refactor", "架构", "architecture", "重写", "rewrite",
    "迁移", "migrate", "全面", "comprehensive", "整体", "entire",
    "所有", "all files", "批量", "batch", "多个文件", "multi-file",
    "设计", "design", "调试", "debug", "排查", "investigate",
    "优化", "optimize", "性能", "performance",
}

COMPLEXITY_MEDIUM_KEYWORDS = {
    "修改", "modify", "fix", "修复", "实现", "implement", "添加", "add",
    "更新", "update", "创建", "create", "搜索", "search", "分析", "analyze",
    "检查", "check", "比较", "compare", "转换", "convert",
}


class ThinkingBudgetManager:
    """Manages adaptive thinking budget for M2.7 models.

    Adjusts thinking budget dynamically based on task complexity
    to optimize for fewer API calls (按次数计费优化).
    """

    def __init__(self, agent: "Agent"):
        self._agent = agent
        self._max_budget: int = 16384
        self._current_budget: int = 16384

    def configure(self, max_budget: int, is_m27: bool) -> None:
        """Configure the manager with agent settings.

        Args:
            max_budget: Maximum thinking budget from config
            is_m27: Whether the model is M2.7
        """
        self._max_budget = max_budget if is_m27 else 0
        self._current_budget = self._max_budget

    def adjust(self, user_message: str) -> None:
        """Adaptively adjust thinking budget based on task complexity.

        按次数计费优化：token免费，思考越深→命中率越高→重试越少→总调用次数越少。
        简单任务给足思考空间也能提高单次完成率。

        Args:
            user_message: The user's message to analyze for complexity
        """
        if not self._agent.is_m27:
            return

        msg_lower = user_message.lower()

        # Detect complexity level from keywords
        high_matches = sum(1 for kw in COMPLEXITY_HIGH_KEYWORDS if kw in msg_lower)
        medium_matches = sum(1 for kw in COMPLEXITY_MEDIUM_KEYWORDS if kw in msg_lower)

        # Estimate file count mentioned
        file_mentions = len(re.findall(
            r'\.(py|js|ts|jsx|tsx|java|go|rs|c|cpp|h|rb|php|yaml|yml|json|toml|md|txt|csv|sql|sh|bash|ps1)',
            msg_lower
        ))

        # Determine complexity
        if high_matches >= 2 or file_mentions >= 4:
            new_budget = THINKING_BUDGET_SUPER
            level = "超复杂"
        elif high_matches >= 1 or file_mentions >= 2 or medium_matches >= 3:
            new_budget = THINKING_BUDGET_COMPLEX
            level = "复杂"
        elif medium_matches >= 1 or file_mentions >= 1:
            new_budget = THINKING_BUDGET_MEDIUM
            level = "中等"
        else:
            new_budget = THINKING_BUDGET_SIMPLE
            level = "简单"

        # Also consider message length as a signal
        msg_tokens = len(user_message) // 3  # rough estimation
        if msg_tokens > 2000:
            new_budget = max(new_budget, THINKING_BUDGET_COMPLEX)
            level = "复杂(长消息)"

        # Constrain to max budget from config
        new_budget = min(new_budget, self._max_budget)

        if new_budget != self._current_budget:
            old_budget = self._current_budget
            self._update_budget(new_budget)
            from ..utils.display import Colors
            print(f"{Colors.DIM}🧠 Thinking budget adjusted: {old_budget} → {new_budget} ({level}任务){Colors.RESET}")

    def _update_budget(self, budget: int) -> None:
        """Update thinking budget via official API.

        Args:
            budget: New thinking budget in tokens
        """
        self._current_budget = budget
        # Update the LLM client's thinking budget via public API
        if hasattr(self._agent.llm, 'configure_thinking_budget'):
            self._agent.llm.configure_thinking_budget(budget)

    @property
    def current_budget(self) -> int:
        """Get current thinking budget."""
        return self._current_budget

    @property
    def max_budget(self) -> int:
        """Get maximum thinking budget."""
        return self._max_budget
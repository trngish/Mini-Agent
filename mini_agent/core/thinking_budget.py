"""Thinking budget management for M2.7 models.

Provides adaptive thinking budget adjustment based on task complexity.
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_context import AgentContext

# Adaptive thinking budget levels (tokens)
# Higher budgets enable deeper reasoning for complex tasks
THINKING_BUDGET_SIMPLE = 16384
THINKING_BUDGET_MEDIUM = 24576
THINKING_BUDGET_COMPLEX = 32768
THINKING_BUDGET_SUPER = 32768

# Complexity indicators for auto-detection
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
    """Manages adaptive thinking budget for M2.7 models.

    Adjusts thinking budget dynamically based on task complexity
    to optimize for fewer API calls (per-call billing optimization).
    """

    def __init__(self, context: "AgentContext"):
        self._context = context
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

        Per-call billing optimization: tokens are free,
        deeper thinking -> higher accuracy -> fewer retries -> fewer total calls.
        Even simple tasks benefit from sufficient thinking space to improve single-call completion rate.

        Args:
            user_message: The user's message to analyze for complexity
        """
        if not self._context.is_m27:
            return

        msg_lower = user_message.lower()

        # Detect complexity level from keywords
        high_matches = sum(1 for kw in COMPLEXITY_HIGH_KEYWORDS if kw in msg_lower)
        medium_matches = sum(1 for kw in COMPLEXITY_MEDIUM_KEYWORDS if kw in msg_lower)

        # Estimate file count mentioned
        file_mentions = len(
            re.findall(
                r"\.(py|js|ts|jsx|tsx|java|go|rs|c|cpp|h|rb|php|yaml|yml|json|toml|md|txt|csv|sql|sh|bash|ps1)",
                msg_lower,
            )
        )

        # Determine complexity
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

        # Also consider message length as a signal
        msg_tokens = len(user_message) // 3  # rough estimation
        if msg_tokens > 2000:
            new_budget = max(new_budget, THINKING_BUDGET_COMPLEX)
            level = "complex(long msg)"

        # Constrain to max budget from config
        new_budget = min(new_budget, self._max_budget)

        if new_budget != self._current_budget:
            old_budget = self._current_budget
            self._update_budget(new_budget)
            from ..utils.display import Colors

            print(f"{Colors.DIM}🧠 Thinking budget adjusted: {old_budget} → {new_budget} ({level} task){Colors.RESET}")

    def _update_budget(self, budget: int) -> None:
        """Update thinking budget via official API.

        Args:
            budget: New thinking budget in tokens
        """
        self._current_budget = budget
        # Sync to context so status displays current value
        self._context.thinking_budget = budget
        # Update the LLM client's thinking budget via public API
        if self._context.llm and hasattr(self._context.llm, "configure_thinking_budget"):
            self._context.llm.configure_thinking_budget(budget)

    @property
    def current_budget(self) -> int:
        """Get current thinking budget."""
        return self._current_budget

    @property
    def max_budget(self) -> int:
        """Get maximum thinking budget."""
        return self._max_budget

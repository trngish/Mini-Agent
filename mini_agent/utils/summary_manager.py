"""Intelligent message summarization with quality-aware triggering.

Adaptive summarization that considers:
- Message complexity (not just token count)
- Task type (simple vs complex)
- Information density (repetitive vs novel)
- Quality score of potential summary
"""

import re
from typing import Any, Literal

from ..schema import Message


# Complexity indicators for adaptive summary targeting
COMPLEXITY_KEYWORDS = {
    'high': {
        'refactor', 'architecture', 'migrate', 'comprehensive', 'entire',
        'all files', 'batch', 'multi-file', 'design', 'debug', 'investigate',
        'optimize', 'performance', '重构', '架构', '迁移',
    },
    'medium': {
        'modify', 'implement', 'add', 'update', 'create', 'search',
        'analyze', 'check', 'compare', 'convert', '修改', '实现',
    },
}

# Repetition patterns that suggest summarization is valuable
REPETITION_PATTERNS = [
    r'重复', r'repeat', r'again', r'same thing',
    r'type the same', r'iterating',
]


class MessageComplexityAnalyzer:
    """Analyzes message complexity for adaptive summarization."""

    @classmethod
    def analyze(cls, messages: list[Message]) -> dict[str, Any]:
        """Analyze message history complexity.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            Dict with complexity metrics:
            - complexity_level: 'low', 'medium', 'high'
            - has_repetition: bool
            - info_density: float (0-1)
            - tool_call_rate: float
        """
        if not messages:
            return {
                'complexity_level': 'low',
                'has_repetition': False,
                'info_density': 0.0,
                'tool_call_rate': 0.0,
            }

        # Count tool calls
        tool_calls = sum(1 for m in messages if m.tool_calls)
        total_messages = len(messages)
        tool_call_rate = tool_calls / max(total_messages, 1)

        # Check for repetition
        all_content = ' '.join(
            m.content if isinstance(m.content, str) else str(m.content)
            for m in messages if m.content
        )
        has_repetition = any(
            re.search(p, all_content, re.IGNORECASE)
            for p in REPETITION_PATTERNS
        )

        # Calculate info density (unique tokens / total tokens)
        words = set(all_content.lower().split())
        total_words = len(all_content.split())
        info_density = len(words) / max(total_words, 1)

        # Complexity level from last user message
        last_user = None
        for m in reversed(messages):
            if m.role == 'user':
                last_user = m.content if isinstance(m.content, str) else str(m.content)
                break

        complexity_level = 'low'
        if last_user:
            last_lower = last_user.lower()
            high_count = sum(1 for kw in COMPLEXITY_KEYWORDS['high'] if kw in last_lower)
            medium_count = sum(1 for kw in COMPLEXITY_KEYWORDS['medium'] if kw in last_lower)
            
            if high_count >= 2 or medium_count >= 4:
                complexity_level = 'high'
            elif high_count >= 1 or medium_count >= 2:
                complexity_level = 'medium'

        return {
            'complexity_level': complexity_level,
            'has_repetition': has_repetition,
            'info_density': info_density,
            'tool_call_rate': tool_call_rate,
        }


class AdaptiveSummaryManager:
    """Manages adaptive summarization with quality awareness."""

    # Summary quality tiers based on message characteristics
    QUALITY_TIERS = {
        'high': {
            'min_budget': 2000,  # tokens
            'preserve_ratio': 0.8,  # preserve 80% of details
            'max_truncation': 500,  # max chars to truncate
        },
        'medium': {
            'min_budget': 1500,
            'preserve_ratio': 0.6,
            'max_truncation': 1000,
        },
        'low': {
            'min_budget': 1000,
            'preserve_ratio': 0.4,
            'max_truncation': 1500,
        },
    }

    def __init__(self, token_limit: int):
        """Initialize summary manager.
        
        Args:
            token_limit: Token limit before summarization is triggered
        """
        self.token_limit = token_limit
        self._complexity_analyzer = MessageComplexityAnalyzer()
        self._last_summary_quality: float = 1.0

    def should_summarize(self, messages: list[Message], estimated_tokens: int, 
                        api_total_tokens: int) -> tuple[bool, str]:
        """Determine if summarization should be triggered.
        
        Args:
            messages: Current message history
            estimated_tokens: Local token estimate
            api_total_tokens: API-reported token count
            
        Returns:
            Tuple of (should_summarize, reason)
        """
        # Check basic threshold
        exceeded = estimated_tokens > self.token_limit or api_total_tokens > self.token_limit
        
        if not exceeded:
            # Only consider early summarization if:
            # 1. High repetition detected
            # 2. Low info density
            # 3. High complexity (might benefit from early compression)
            complexity = self._complexity_analyzer.analyze(messages[-10:])  # Last 10 msgs
            
            early_trigger = (
                complexity['has_repetition'] or
                complexity['info_density'] < 0.3 or
                (complexity['complexity_level'] == 'high' and complexity['tool_call_rate'] > 0.5)
            )
            
            if early_trigger:
                return True, f"early_trigger:{complexity['complexity_level']}"
            return False, "below_threshold"

        # Token exceeded - determine summary quality tier
        complexity = self._complexity_analyzer.analyze(messages)
        
        if complexity['has_repetition'] or complexity['info_density'] < 0.4:
            tier = 'low'
        elif complexity['complexity_level'] == 'high':
            tier = 'high'
        else:
            tier = 'medium'
        
        return True, f"threshold_exceeded:tier={tier}"

    def get_summary_config(self, tier: str) -> dict:
        """Get summarization configuration for quality tier.
        
        Args:
            tier: 'high', 'medium', or 'low'
            
        Returns:
            Summary configuration dict
        """
        return self.QUALITY_TIERS.get(tier, self.QUALITY_TIERS['medium'])

    def estimate_summary_quality(self, original: list[Message], summary: str) -> float:
        """Estimate quality of generated summary.
        
        Args:
            original: Original messages
            summary: Generated summary text
            
        Returns:
            Quality score 0-1
        """
        if not summary:
            return 0.0

        # Check coverage - does summary mention key actions?
        original_content = ' '.join(
            str(m.content) for m in original if m.content
        )
        
        # Key terms to check coverage
        key_terms = ['tool', 'call', 'result', 'error', 'success', 'file', 'read', 'write']
        coverage = sum(1 for term in key_terms if term in original_content.lower()) / len(key_terms)
        
        # Length ratio (should be much shorter but not too short)
        original_len = len(original_content)
        summary_len = len(summary)
        ratio = summary_len / max(original_len, 1)
        
        # Ideal ratio is 0.1-0.3
        length_score = 1.0 if 0.1 <= ratio <= 0.3 else max(0, 1 - abs(ratio - 0.2) * 2)
        
        # Combined score
        quality = coverage * 0.6 + length_score * 0.4
        self._last_summary_quality = quality
        
        return quality

    def should_skip_next_check(self, last_summary_quality: float) -> bool:
        """Determine if next token check should be skipped.
        
        Skip after high-quality summary to avoid consecutive triggers.
        """
        return last_summary_quality > 0.7
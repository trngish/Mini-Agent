"""MiniMax M2.7 specific optimizations and configurations.

This module provides model-specific optimizations for MiniMax M2.7 model,
including extended thinking, context management, and tool calling optimizations.

M2.7 Specifications (from https://www.minimaxi.com/models/text/m27):
- Context Window: 1M (1,000,000) tokens
- Max Output: 32K (32,768) tokens
- Extended Thinking: Up to 32K tokens budget
- Parallel Tool Calls: Native support (97% following rate on 40 complex skills)
- Agent Teams: Native multi-agent collaboration capability

Note: M27Config has been moved to model_utils.py as is_m27_model().
      Use model_utils.get_model_specs() for model specifications.
"""

from typing import Any

# Re-export from model_utils for backwards compatibility
from .model_utils import (
    is_m27_model as is_m27_enabled,
    get_model_specs,
    M27_MODEL_IDENTIFIERS,
    MINIMAX_MODEL_IDENTIFIERS,
)
from .model_utils import ModelSpecs


class M27PromptOptimizer:
    """Optimizes prompts for M2.7 model.

    M2.7 has the following specifications:
    - Context Window: 1M (1,000,000) tokens
    - Max Output: 32K (32,768) tokens
    - Extended Thinking: Up to 32K tokens budget
    - Native Agent Teams support
    """

    # System prompt template for M2.7 that emphasizes its capabilities
    M27_SYSTEM_PROMPT_TEMPLATE = """You are Mini-Agent powered by MiniMax M2.7, an advanced AI assistant with extended reasoning capabilities.

## M2.7 Capabilities:
- **Extended Thinking**: You can use internal reasoning (up to 32K tokens) to break down complex problems
- **Large Context**: You can handle conversations with up to 1M tokens context window
- **Tool Use**: You can use various tools to complete tasks (parallel tool calls supported, 97% following rate)
- **Code Execution**: You can execute code and analyze results
- **Agent Teams**: Native multi-agent collaboration with role boundary and adversarial reasoning

## M2.7 Benchmark Performance:
- SWE-Pro: 56.22% (comparable to GPT-5.3-Codex)
- Terminal Bench 2: 57.0%
- GDPval-AA ELO: 1500
- Toolathon: 46.3% (global top tier)

## Best Practices:
1. For complex tasks (software engineering, analysis), use step-by-step reasoning with extended thinking (16K+ tokens)
2. Use parallel tool calls for independent operations (up to 5 concurrent)
3. Leverage 1M context for complex multi-file operations
4. For skills-heavy tasks (>2000 tokens), M2.7 maintains 97% following rate
5. Be concise but thorough in your responses

{ADDITIONAL_INSTRUCTIONS}

## Current Workspace
You are currently working in: `{WORKSPACE_DIR}`
All relative paths will be resolved relative to this directory.
"""

    @classmethod
    def optimize_system_prompt(cls, base_prompt: str, workspace_dir: str) -> str:
        """Optimize system prompt for M2.7.

        Args:
            base_prompt: The base system prompt
            workspace_dir: Current workspace directory

        Returns:
            Optimized system prompt
        """
        # If the prompt already mentions extended thinking or M2.7, don't modify
        if "M2.7" in base_prompt or "extended thinking" in base_prompt.lower():
            return base_prompt

        # Format the M2.7 optimized template
        additional_instructions = ""

        # Check if base prompt has useful content
        if base_prompt.strip():
            additional_instructions = base_prompt.strip()

        optimized = cls.M27_SYSTEM_PROMPT_TEMPLATE.format(
            ADDITIONAL_INSTRUCTIONS=additional_instructions,
            WORKSPACE_DIR=workspace_dir,
        )

        return optimized

    @classmethod
    def get_thinking_block(cls, content: str) -> dict[str, Any]:
        """Create an extended thinking block for M2.7.

        Args:
            content: The thinking content

        Returns:
            Thinking block in M2.7 format
        """
        return {
            "type": "thinking",
            "thinking": content,
        }

    @classmethod
    def extract_thinking_from_response(cls, response: Any) -> str | None:
        """Extract thinking content from M2.7 response.

        Args:
            response: The LLM response object

        Returns:
            Thinking content if present, None otherwise
        """
        # For M2.7, thinking is in the thinking field of the response
        if hasattr(response, 'thinking') and response.thinking:
            return response.thinking

        # Check content blocks
        if hasattr(response, 'content') and isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    return block.get("thinking")
                if hasattr(block, "type") and block.type == "thinking":
                    return getattr(block, "thinking", None)

        return None


class M27ContextManager:
    """Manages context window optimization for M2.7.

    M2.7 has a 1M (1,000,000) token context window, so we can process
    significantly more messages before needing summarization.
    """

    def __init__(self, token_limit: int = 800_000):
        """Initialize context manager.

        Args:
            token_limit: Token limit before summarization (default: 800K for M2.7)
                        This is 80% of the 1M context window to leave room
                        for system prompt and tool schemas.
        """
        self.token_limit = token_limit
        self.api_total_tokens = 0

    def should_summarize(self, estimated_tokens: int) -> bool:
        """Determine if summarization should be triggered.

        Args:
            estimated_tokens: Estimated token count from local calculation

        Returns:
            True if summarization should be triggered
        """
        # Trigger if either local estimate or API tokens exceed limit
        return estimated_tokens > self.token_limit or self.api_total_tokens > self.token_limit

    def get_optimized_summarization_prompt(self, messages: list[Any]) -> str:
        """Generate an optimized summarization prompt for M2.7.

        Args:
            messages: List of messages to summarize

        Returns:
            Optimized summarization prompt
        """
        return """You are summarizing an Agent execution history for MiniMax M2.7.

Your task is to create a concise summary that preserves:
1. Key decisions and reasoning
2. Tool calls and their purposes
3. Important results and findings
4. Current state of the task

Requirements:
- Keep within 2000 words (M2.7 has large context, but summaries should be concise)
- Focus on actionable information
- Preserve context for continuing the task
- Use English for the summary
- Structure: Round X: [what was done] -> [results]
"""

    def calculate_optimal_batch_size(self, message_count: int) -> int:
        """Calculate optimal batch size for processing messages.

        M2.7's large context window allows processing more messages at once.

        Args:
            message_count: Total number of messages

        Returns:
            Optimal batch size
        """
        # For M2.7 with 1M context, we can process many more messages at once
        if message_count <= 100:
            return message_count
        elif message_count <= 500:
            return 100
        elif message_count <= 2000:
            return 200
        else:
            return 500  # M2.7 can handle larger batches


class M27ToolOptimizer:
    """Optimizes tool calling for M2.7 model."""

    # Tool categories that M2.7 handles well in parallel
    PARALLEL_COMPATIBLE_TOOLS = {
        "bash",  # Independent bash commands
        "read",  # Reading multiple files
        "grep",  # Multiple grep searches
        "find",  # File searches
    }

    # Tools that should always be sequential
    SEQUENTIAL_TOOLS = {
        "write",  # Write operations depend on previous state
        "edit",  # Edit operations depend on file state
        "git",  # Git operations are inherently sequential
    }

    # M2.7 benchmark: 97% skills following rate on 40 complex skills (>2000 tokens each)
    COMPLEX_SKILLS_THRESHOLD = 2000  # tokens

    @classmethod
    def can_parallelize(cls, tool_names: list[str]) -> bool:
        """Check if a list of tool calls can be parallelized.

        Args:
            tool_names: List of tool names

        Returns:
            True if all tools can be run in parallel
        """
        # If any tool is sequential, don't parallelize
        for name in tool_names:
            if name in cls.SEQUENTIAL_TOOLS:
                return False

        # Check if all tools are in the parallel-compatible set
        return all(name in cls.PARALLEL_COMPATIBLE_TOOLS for name in tool_names)

    @classmethod
    def optimize_tool_schema(cls, tool_schema: dict[str, Any]) -> dict[str, Any]:
        """Optimize tool schema for M2.7.

        Args:
            tool_schema: The tool schema

        Returns:
            Optimized schema
        """
        # M2.7 handles complex schemas well, but we can add hints
        optimized = tool_schema.copy()

        # Add descriptions that help M2.7 understand tool purpose
        if "description" not in optimized or not optimized["description"]:
            optimized["description"] = f"Tool: {optimized.get('name', 'unknown')}"

        return optimized


class M27AgentTeams:
    """Agent Teams support for M2.7.

    M2.7 has native Agent Teams capability with:
    - Role boundary preservation
    - Adversarial reasoning
    - Protocol following
    - Behavioral differentiation
    """

    # Agent team roles supported by M2.7
    SUPPORTED_ROLES = {
        "planner",      # Task planning and decomposition
        "executor",     # Tool execution and implementation
        "reviewer",     # Quality assurance and validation
        "coordinator",  # Multi-agent coordination
    }

    # Agent team protocols
    SUPPORTED_PROTOCOLS = {
        "sequential",  # One agent at a time
        "parallel",    # Multiple agents simultaneously
        "hierarchical", # Manager-subordinate structure
    }

    @classmethod
    def create_team_prompt(cls, roles: list[str], task: str) -> str:
        """Create a prompt for multi-agent collaboration.

        Args:
            roles: List of agent roles
            task: The task to be accomplished

        Returns:
            Team collaboration prompt
        """
        role_str = ", ".join(roles)
        return f"""You are coordinating a team of agents with roles: {role_str}

Task: {task}

Team Guidelines:
1. Each agent should maintain its role boundary
2. Use adversarial reasoning to challenge assumptions
3. Follow established protocols for communication
4. Differentiate behaviors based on role responsibilities

M2.7 supports native Agent Teams with these capabilities built-in.
"""
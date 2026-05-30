"""MiniMax M2.7特定优化和配置。

此模块为MiniMax M2.7模型提供特定优化，包括扩展思考、上下文管理
和工具调用优化。

M2.7规格（来自 https://www.minimaxi.com/models/text/m27）：
- 上下文窗口：1M（1,000,000）tokens
- 最大输出：32K（32,768）tokens
- 扩展思考：最高32K tokens预算
- 并行工具调用：原生支持（40项复杂技能中97%的遵循率）
- Agent Teams：原生多Agent协作能力

注意：M27Config已移至model_utils.py中的is_m27_model()。
      使用model_utils.get_model_specs()获取模型规格。
"""

from typing import Any

# 为保持向后兼容性，从model_utils重新导出


class M27PromptOptimizer:
    """为M2.7模型优化提示词。

    M2.7具有以下规格：
    - 上下文窗口：1M（1,000,000）tokens
    - 最大输出：32K（32,768）tokens
    - 扩展思考：最高32K tokens预算
    - 原生Agent Teams支持
    """

    # M2.7的系统提示模板，强调其能力
    M27_SYSTEM_PROMPT_TEMPLATE = """\
You are Mini-Agent powered by MiniMax M2.7, an advanced AI assistant with extended reasoning capabilities.

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
        """为M2.7优化系统提示。

        Args:
            base_prompt: 基础系统提示
            workspace_dir: 当前工作区目录

        Returns:
            优化后的系统提示
        """
        # 如果提示已提及扩展思考或M2.7，则不修改
        if "M2.7" in base_prompt or "extended thinking" in base_prompt.lower():
            return base_prompt

        # 格式化M2.7优化模板
        additional_instructions = ""

        # 检查基础提示是否有有用内容
        if base_prompt.strip():
            additional_instructions = base_prompt.strip()

        optimized = cls.M27_SYSTEM_PROMPT_TEMPLATE.format(
            ADDITIONAL_INSTRUCTIONS=additional_instructions,
            WORKSPACE_DIR=workspace_dir,
        )

        return optimized

    @classmethod
    def get_thinking_block(cls, content: str) -> dict[str, Any]:
        """为M2.7创建扩展思考块。

        Args:
            content: 思考内容

        Returns:
            M2.7格式的思考块
        """
        return {
            "type": "thinking",
            "thinking": content,
        }

    @classmethod
    def extract_thinking_from_response(cls, response: Any) -> str | None:
        """从M2.7响应中提取思考内容。

        Args:
            response: LLM响应对象

        Returns:
            如果存在思考内容则返回，否则返回None
        """
        # 对于M2.7，思考内容在响应的thinking字段中
        if hasattr(response, "thinking") and response.thinking:
            return response.thinking  # type: ignore[no-any-return]

        # 检查content块
        if hasattr(response, "content") and isinstance(response.content, list):
            for block in response.content:
                if isinstance(block, dict) and block.get("type") == "thinking":
                    return block.get("thinking")
                if hasattr(block, "type") and block.type == "thinking":
                    return getattr(block, "thinking", None)

        return None


class M27ContextManager:
    """管理M2.7的上下文窗口优化。

    M2.7具有1M（1,000,000）token的上下文窗口，因此可以在需要
    摘要之前处理更多的消息。
    """

    def __init__(self, token_limit: int = 800_000):
        """初始化上下文管理器。

        Args:
            token_limit: 摘要前的token限制（M2.7默认为800K）
                        这是1M上下文窗口的80%，为系统提示
                        和工具模式留出空间。
        """
        self.token_limit = token_limit
        self.api_total_tokens = 0

    def should_summarize(self, estimated_tokens: int) -> bool:
        """判断是否应触发摘要。

        Args:
            estimated_tokens: 从本地计算估计的token数

        Returns:
            如果应触发摘要则返回True
        """
        # 如果本地估计或API tokens超过限制，则触发
        return estimated_tokens > self.token_limit or self.api_total_tokens > self.token_limit

    def get_optimized_summarization_prompt(
        self,
        _messages: list[Any],  # noqa: ARG002
    ) -> str:
        """为M2.7生成优化的摘要提示。

        Args:
            messages: 要摘要的消息列表

        Returns:
            优化的摘要提示
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
        """计算处理消息的最佳批量大小。

        M2.7的大上下文窗口允许一次处理更多消息。

        Args:
            message_count: 消息总数

        Returns:
            最佳批量大小
        """
        # 对于具有1M上下文的M2.7，我们可以一次处理更多消息
        if message_count <= 100:
            return message_count
        elif message_count <= 500:
            return 100
        elif message_count <= 2000:
            return 200
        else:
            return 500  # M2.7可以处理更大的批次


class M27ToolOptimizer:
    """为M2.7模型优化工具调用。"""

    # M2.7能良好并行处理的工具类别
    PARALLEL_COMPATIBLE_TOOLS = {
        "bash",  # 独立的bash命令
        "read",  # 读取多个文件
        "grep",  # 多个grep搜索
        "find",  # 文件搜索
    }

    # 应该始终顺序执行的工具
    SEQUENTIAL_TOOLS = {
        "write",  # 写操作依赖先前状态
        "edit",  # 编辑操作依赖文件状态
        "git",  # Git操作本质上是顺序的
    }

    # M2.7基准测试：40项复杂技能（每个>2000 tokens）中97%的技能遵循率
    COMPLEX_SKILLS_THRESHOLD = 2000  # tokens

    @classmethod
    def can_parallelize(cls, tool_names: list[str]) -> bool:
        """检查工具调用列表是否可以并行化。

        Args:
            tool_names: 工具名称列表

        Returns:
            如果所有工具都可以并行运行则返回True
        """
        # 如果任何工具是顺序的，则不并行化
        for name in tool_names:
            if name in cls.SEQUENTIAL_TOOLS:
                return False

        # 检查所有工具是否都在并行兼容集中
        return all(name in cls.PARALLEL_COMPATIBLE_TOOLS for name in tool_names)

    @classmethod
    def optimize_tool_schema(cls, tool_schema: dict[str, Any]) -> dict[str, Any]:
        """为M2.7优化工具模式。

        Args:
            tool_schema: 工具模式

        Returns:
            优化后的模式
        """
        # M2.7能很好地处理复杂模式，但我们可以添加提示
        optimized = tool_schema.copy()

        # 添加帮助M2.7理解工具用途的描述
        if "description" not in optimized or not optimized["description"]:
            optimized["description"] = f"Tool: {optimized.get('name', 'unknown')}"

        return optimized


class M27AgentTeams:
    """M2.7的Agent Teams支持。

    M2.7具有原生Agent Teams能力，包括：
    - 角色边界保持
    - 对抗性推理
    - 协议遵循
    - 行为差异化
    """

    # M2.7支持的Agent团队角色
    SUPPORTED_ROLES = {
        "planner",  # 任务规划和分解
        "executor",  # 工具执行和实现
        "reviewer",  # 质量保证和验证
        "coordinator",  # 多Agent协调
    }

    # Agent团队协议
    SUPPORTED_PROTOCOLS = {
        "sequential",  # 一次一个Agent
        "parallel",  # 多个Agent同时执行
        "hierarchical",  # 管理者-从属结构
    }

    @classmethod
    def create_team_prompt(cls, roles: list[str], task: str) -> str:
        """创建多Agent协作提示。

        Args:
            roles: Agent角色列表
            task: 要完成的任务

        Returns:
            团队协作提示
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

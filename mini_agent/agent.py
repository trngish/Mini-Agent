"""核心Agent实现"""

from __future__ import annotations

# 硬性规则：所有Git操作需要用户明确同意
# 包括但不限于：git add, git commit, git push, git merge
# 违反将被视为未授权
import asyncio
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from .core.agent_context import AgentContext
from .core.approval import ApprovalManager
from .core.error_recovery import ErrorRecoveryManager
from .core.execution_engine import ExecutionEngine
from .core.health_check import HealthChecker
from .core.self_healing import SelfHealingManager
from .core.message_manager import MessageManager
from .core.metrics import PerformanceMetrics
from .core.rate_limiter import RateLimiter
from .core.retry_handler import create_retry_handler
from .core.semantic_memory import SemanticMemory
from .core.step_runner import StepRunner
from .core.thinking_budget import ThinkingBudgetManager
from .core.token_tracker import TokenTracker
from .llm import LLMClient
from .logger import AgentLogger
from .schema import AgentMode, Message, ToolCall
from .schema.schema import WRITE_TOOLS
from .session import SessionManager
from .subagent import SubAgentResult
from .tools.base import Tool
from .utils import Colors
from .utils.context_cache import create_cache_for_workspace
from .utils.error_handler import LLMErrorClassifier, format_llm_error
from .utils.model_utils import get_token_limit_for_model, is_m27_model
from .utils.task_state import get_task_manager
from .utils.thinking_manager import ThinkingManager

logger = logging.getLogger(__name__)

# 常量 - 避免魔法数字
STREAM_BUFFER_SIZE = int(os.environ.get("MINI_AGENT_STREAM_BUFFER_SIZE", "8"))
DEFAULT_ENCODING_NAME = "cl100k_base"


class Agent:
    """单Agent，配备基础工具和MCP支持。"""

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        tools: list[Tool],
        max_steps: int = 50,
        workspace_dir: str = "./workspace",
        token_limit: int = 80000,
        m27_config: dict[str, Any] | None = None,
        mode: AgentMode = AgentMode.YOLO,
    ):
        # 首先存储模式，因为它在创建上下文时会用到
        self.mode = mode
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.tool_list = list(tools)
        self.max_steps = max_steps
        self.workspace_dir = Path(workspace_dir)
        self.cancel_event: asyncio.Event | None = None
        self._session_manager = SessionManager(workspace_dir=self.workspace_dir)
        self.write_tools = WRITE_TOOLS

        # M2.7 特定配置
        self.m27_config = m27_config or {}
        model_name = getattr(llm_client, "model", "")
        self.is_m27 = is_m27_model(model_name)

        # 使用统一的token限制计算
        self.token_limit = get_token_limit_for_model(
            model_name, self.m27_config.get("token_limit") if self.is_m27 else token_limit
        )

        # M2.7 支持最高32K输出tokens
        self.max_output_tokens = self.m27_config.get("max_output_tokens", 16384) if self.is_m27 else 8192

        # 配置文件中的最大预算
        self._max_thinking_budget = self.m27_config.get("thinking_budget_tokens", 16384) if self.is_m27 else 0
        self.thinking_budget = self._max_thinking_budget

        # 优化：并行工具执行的批处理大小
        self._max_tools_per_call = self.m27_config.get("max_concurrent_tools", 20) if self.is_m27 else 3

        # 确保工作目录存在
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # 将工作目录信息注入系统提示词
        if "Current Workspace" not in system_prompt:
            workspace_info = (
                f"\n\n## Current Workspace\n"
                f"You are currently working in: `{self.workspace_dir.absolute()}`"
                f"\nAll relative paths will be resolved relative to this directory."
            )
            system_prompt = system_prompt + workspace_info

        self.system_prompt = system_prompt

        # D11修复：跨会话知识持久化的语义内存
        self._semantic_memory = SemanticMemory(workspace_dir=self.workspace_dir)

        # 将跨会话记忆注入系统提示词
        memory_context = self._semantic_memory.get_context_for_injection(max_entries=6)
        if memory_context:
            self.system_prompt += "\n" + memory_context

        # 将模式指令注入系统提示词
        mode_instructions = {
            AgentMode.PLAN: (
                "\n\n## Mode: Plan\nYou are in PLAN mode. You can ONLY read files and explore."
                " You MUST NOT modify files, execute commands, or make any changes."
                " Propose a plan to the user before any write operations."
            ),
            AgentMode.AGENT: (
                "\n\n## Mode: Agent\nYou are in AGENT mode (default)."
                " You can use all tools. Each tool call will require user approval."
            ),
            AgentMode.YOLO: (
                "\n\n## Mode: YOLO\nYou are in YOLO mode. All tool calls are auto-approved. Execute efficiently."
            ),
        }
        self.system_prompt += mode_instructions.get(self.mode, "")

        # 初始化日志记录器
        self.logger = AgentLogger()

        # 带有增量估算的token跟踪器
        self._token_tracker = TokenTracker()

        # 每个步骤后自动保存会话
        self.auto_save = os.environ.get("MINI_AGENT_AUTO_SAVE", "true").lower() == "true"

        # 首先创建AgentContext - 这是状态的唯一真实来源
        # 所有核心模块接收context的引用而不是Agent（打破循环依赖）
        self._context = AgentContext(
            messages=[Message(role="system", content=self.system_prompt)],
            mode=self.mode,
            max_steps=self.max_steps,
            workspace_dir=self.workspace_dir,
            token_limit=self.token_limit,
            api_call_count=0,
            api_total_tokens=0,
            is_m27=self.is_m27,
            thinking_budget=self._max_thinking_budget,
            llm=self.llm,
            auto_save=self.auto_save,
            token_tracker=self._token_tracker,
            record_context_fn=self._record_context_internal,
        )

        # 现在使用AgentContext初始化核心模块（无循环依赖）
        self._thinking_budget_manager = ThinkingBudgetManager(self._context)
        self._thinking_budget_manager.configure(self._max_thinking_budget, self.is_m27)

        self._error_recovery = ErrorRecoveryManager(self._context)
        self._metrics = PerformanceMetrics(self._context)
        self._retry_handler = create_retry_handler(self)
        self._approval_manager = ApprovalManager(mode=mode, write_tools=self.write_tools)
        self._health_checker = HealthChecker(self._context)
        self._rate_limiter = RateLimiter()

        # 通过委托方法访问错误恢复和指标

        # 性能优化的流式缓冲
        self._stream_buffer_thinking: list[str] = []
        self._stream_buffer_text: list[str] = []
        self._buffer_flush_threshold = STREAM_BUFFER_SIZE * 2

        # 执行引擎
        self._execution_engine = ExecutionEngine(
            tools=self.tools,
            logger=self.logger,
            retry_handler=self._retry_handler,
            metrics=self._metrics,
            error_recovery=self._error_recovery,
            write_tools=self.write_tools,
            rate_limiter=self._rate_limiter,
        )

        # 用于减少冗余文件读取/搜索的上下文缓存
        # D14修复：工作区隔离缓存 - 每个工作区拥有自己的缓存实例
        self._context_cache = create_cache_for_workspace(self.workspace_dir)

        # 预热缓存：加载频繁访问的文件
        if os.environ.get("MINI_AGENT_CACHE_WARMUP", "true").lower() == "true":
            try:
                cached_count = self._context_cache.warmup(self.workspace_dir)
                if cached_count > 0:
                    print(f"{Colors.DIM}📦 缓存已预热，加载了 {cached_count} 个文件{Colors.RESET}")
            except Exception as e:
                logger.debug("缓存预热失败: %s", e)

        # 用于摘要的消息管理器
        self._message_manager = MessageManager(self.token_limit)

        # 思考管理器：防止因截断思考导致上下文溢出
        self._thinking_manager: ThinkingManager | None = None
        if self.is_m27:
            self._thinking_manager = ThinkingManager(max_thinking_tokens=80_000)

        # 性能：限制健康检查频率
        self._last_health_check_step = -1
        self._health_check_interval = 5
        self._loop_detection_streak = 0

        # 为会话持久化存储上次运行结果
        self._last_result: str | None = None
        # 为会话持久化存储分析结果
        self._last_analysis: str | None = None

        # 自愈引擎：检测异常并自动修复源代码。
        # 自动检测源代码目录：检查mini_agent包是否在此工作区中。
        _heal_source = self.workspace_dir
        if (_heal_source / "mini_agent" / "agent.py").exists():
            pass  # 工作区就是项目根目录（可编辑安装情况）
        elif (_heal_source.parent / "mini_agent" / "agent.py").exists():
            _heal_source = _heal_source.parent  # 工作区是一个子目录
        else:
            # 回退：使用包安装目录
            from pathlib import Path as P
            _heal_source = P(__file__).parent.parent
        self._self_healing = SelfHealingManager(
            source_dir=_heal_source,
            llm_client=self.llm if os.environ.get("MINI_AGENT_AUTO_HEAL", "0") == "1" else None,
        )

    def _record_context_internal(self, content: str, category: str = "auto") -> None:
        """AgentContext内部方法，用于记录上下文。"""
        self.record_context(content, category)

    def add_user_message(self, content: str) -> None:
        """将用户消息添加到历史记录中。"""
        self._context.add_message(Message(role="user", content=content))
        # 根据任务复杂度自适应调整思考预算
        self._thinking_budget_manager.adjust(content)

    @property
    def messages(self) -> list[Message]:
        """通过AgentContext获取消息历史。"""
        return self._context.get_messages()

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        """替换消息历史（向后兼容赋值）。"""
        self.replace_messages(value)

    def append_message(self, message: Message) -> None:
        """通过AgentContext追加消息到历史。"""
        self._context.add_message(message)

    def replace_messages(self, messages: list[Message]) -> None:
        """通过AgentContext替换整个消息历史。"""
        self._context.set_messages(messages)

    # api_call_count的向后兼容属性
    @property
    def api_call_count(self) -> int:
        """通过AgentContext获取API调用次数。"""
        return self._context.api_call_count

    @api_call_count.setter
    def api_call_count(self, value: int) -> None:
        """通过AgentContext设置API调用次数。"""
        self._context.api_call_count = value

    # api_total_tokens的向后兼容属性
    @property
    def api_total_tokens(self) -> int:
        """通过AgentContext获取API总token数。"""
        return self._context.api_total_tokens

    @api_total_tokens.setter
    def api_total_tokens(self, value: int) -> None:
        """通过AgentContext设置API总token数。"""
        self._context.api_total_tokens = value

    @property
    def consecutive_failures(self) -> int:
        """通过AgentContext获取连续失败次数。"""
        return self._context.consecutive_failures

    @consecutive_failures.setter
    def consecutive_failures(self, value: int) -> None:
        """通过AgentContext设置连续失败次数。"""
        self._context.consecutive_failures = value

    def record_context(self, content: str, category: str = "auto") -> None:
        """自动记录重要上下文，无需显式工具调用。

        当Agent遇到值得记住的重要信息时，会在内部调用此方法。

        Args:
            content: 要记录的上下文
            category: 类别标签（默认："auto"用于自动记录）
        """
        # 如果可用，使用会话笔记工具
        note_tool = self.tools.get("record_note")
        if note_tool:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(note_tool.execute(content=content, category=category))
            except RuntimeError:
                self.logger.debug("无运行中的事件循环，无法进行后台记录")
            except Exception as e:
                self.logger.debug(f"后台记录失败: {e}")

    def _check_cancelled(self) -> bool:
        """检查Agent执行是否已取消。

        Returns:
            如果已取消则返回True，否则返回False。
        """
        return bool(self.cancel_event is not None and self.cancel_event.is_set())

    def _cleanup_incomplete_messages(self) -> None:
        """删除不完整的助手消息及其部分工具结果。

        这确保了在取消后通过仅删除当前步骤的不完整消息
        来保持消息一致性，保留已完成的步骤。
        """
        # 找到最后一条助手消息的索引
        last_assistant_idx = -1
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx == -1:
            # 没有找到助手消息，无需清理
            return

        # 删除最后一条助手消息及其之后的所有工具结果
        removed_count = len(self.messages) - last_assistant_idx
        if removed_count > 0:
            self.replace_messages(self.messages[:last_assistant_idx])
            print(f"{Colors.DIM}   已清理 {removed_count} 条未完成消息{Colors.RESET}")

    async def _summarize_messages(self) -> None:
        """消息历史摘要：委托给MessageManager处理。"""
        old_token_count = self._context.api_total_tokens
        new_messages = await self._message_manager.summarize_messages(self.messages, self.api_total_tokens, logger)
        if new_messages is not self.messages:
            self.replace_messages(new_messages)
            self._token_tracker.invalidate_cache()
            # 自愈：当摘要触发时记录token压力
            self._self_healing.record_anomaly(
                "token_pressure", 0.5,
                {"api_tokens": old_token_count, "limit": self.token_limit}, 0,
            )

    async def run(self, cancel_event: asyncio.Event | None = None) -> str:
        """执行agent循环直到任务完成或达到最大步数。

        Args:
            cancel_event: 可选的asyncio.Event，可设置以取消执行。
                          设置后，agent将在下一个安全检查点停止
                         （在当前步骤完成后停止以保持消息一致）。

        Returns:
            最终响应内容，或错误消息（包括取消消息）。
        """
        # 设置取消事件（也可以在调用run()之前通过self.cancel_event设置）
        if cancel_event is not None:
            self.cancel_event = cancel_event

        # 开始新运行，初始化日志文件
        self.logger.start_new_run()
        print(f"{Colors.DIM}📝 日志文件: {self.logger.get_log_file_path()}{Colors.RESET}")

        # 初始化指标会话跟踪
        self._metrics.set_session_id(str(id(self)))

        step = 0
        run_start_time = perf_counter()
        step_runner = StepRunner(self, run_start_time)
        self._loop_detection_streak = 0

        task_mgr = get_task_manager()
        if task_mgr.current_task:
            task_mgr.end_task()
        last_user = next(
            (m.content for m in reversed(self.messages) if m.role == "user" and m.content),
            "agent-run",
        )
        task_mgr.start_task(str(id(self)), last_user[:200], max_steps=self.max_steps)

        while step < self.max_steps:
            task_mgr.increment_steps()

            # 自愈：每个步骤递增并衰减异常分数
            self._self_healing.tick(step)
            should_heal, heal_reason = self._self_healing.should_heal(step)
            if should_heal:
                await self._trigger_self_healing(step, heal_reason)

            should_stop, stop_reason = task_mgr.check_should_stop()
            if should_stop:
                stop_msg = f"任务已停止: {stop_reason}"
                self._last_result = stop_msg
                print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {stop_msg}{Colors.RESET}")
                return stop_msg

            # 在每个步骤开始时检查取消
            if self._check_cancelled():
                # D1修复：在取消前刷新任何缓冲的流式内容。
                # 没有这个，部分流式的助手响应内容会
                # 丢失，用户永远看不到AI将要说的内容。
                if self._stream_buffer_text:
                    print("".join(self._stream_buffer_text), end="", flush=True)
                    self._stream_buffer_text = []
                if self._stream_buffer_thinking:
                    print(f"{Colors.DIM}{''.join(self._stream_buffer_thinking)}{Colors.RESET}", end="", flush=True)
                    self._stream_buffer_thinking = []
                self._cleanup_incomplete_messages()
                cancel_msg = "任务已被用户取消。"
                self._last_result = cancel_msg
                # E4修复：记录取消以便审计追踪
                logger.info("任务已被用户取消")
                print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {cancel_msg}{Colors.RESET}")
                return cancel_msg

            step_start_time = perf_counter()
            # 检查并摘要消息历史以防止上下文溢出
            await self._summarize_messages()

            # 步骤头部 - 统一的单次打印以提高性能
            step_text = f"{Colors.BOLD}{Colors.BRIGHT_CYAN}Step {step + 1}/{self.max_steps}{Colors.RESET}"
            box_width = 44
            pad = box_width - len(f"  Step {step + 1}/{self.max_steps}") - 1
            print(
                f"\n  {Colors.DIM}╭{'─' * box_width}╮{Colors.RESET}\n"
                f"  {Colors.DIM}│{Colors.RESET}  {step_text}{' ' * max(0, pad)}{Colors.DIM}│{Colors.RESET}\n"
                f"  {Colors.DIM}╰{'─' * box_width}╯{Colors.RESET}"
            )

            # 获取LLM调用的工具列表（在会话期间缓存）
            tool_list = self.tool_list

            # 记录LLM请求并直接使用Tool对象调用LLM
            self.logger.log_request(messages=self.messages, tools=tool_list)

            try:
                # 跟踪流式状态以保证正确排序
                thinking_started = False
                text_pending: list[str] = []

                def _flush_thinking_buffer() -> None:
                    """刷新缓冲的思考输出。"""
                    if self._stream_buffer_thinking:
                        output = "".join(self._stream_buffer_thinking)
                        print(f"{Colors.DIM}{output}{Colors.RESET}", end="", flush=True)
                        self._stream_buffer_thinking = []

                def _flush_text_buffer() -> None:
                    """刷新缓冲的文本输出。"""
                    if self._stream_buffer_text:
                        output = "".join(self._stream_buffer_text)
                        print(output, end="", flush=True)
                        self._stream_buffer_text = []

                def on_thinking(text: str) -> None:
                    nonlocal thinking_started, text_pending  # noqa: B023
                    if not thinking_started:
                        print(f"\n  {Colors.BOLD}{Colors.MAGENTA}🧠 思考{Colors.RESET}")
                        thinking_started = True
                    # 修复：将text_pending移到text_buffer但不刷新。
                    # 文本必须等待所有思考完成后才能显示。
                    while text_pending:  # noqa: B023
                        pending = text_pending.pop(0)  # noqa: B023
                        self._stream_buffer_text.append(pending)
                    # 流式思考内容
                    self._stream_buffer_thinking.append(text)
                    if len(self._stream_buffer_thinking) >= self._buffer_flush_threshold:
                        _flush_thinking_buffer()

                def on_text(text: str) -> None:  # noqa: B023
                    nonlocal thinking_started, text_pending  # noqa: B023
                    if thinking_started:  # noqa: B023
                        # 修复：在思考进行时静默累积文本。
                        # 文本只会在思考完全完成后显示。
                        self._stream_buffer_text.append(text)
                    else:
                        text_pending.append(text)  # noqa: B023

                response = await self.llm.generate(
                    messages=self.messages,
                    tools=tool_list,
                    on_text=on_text,
                    on_thinking=on_thinking,
                )

                # 打印助手头部。首先刷新思考，然后刷新文本。
                # 这确保用户在看到任何结果文本之前看到所有思考内容。
                if thinking_started:
                    # 将剩余的text_pending排入缓冲区（静默）
                    while text_pending:
                        pending = text_pending.pop(0)
                        self._stream_buffer_text.append(pending)
                    # 首先刷新所有思考内容
                    _flush_thinking_buffer()
                    # 然后打印头部并刷新所有累积的文本
                    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🤖 助手:{Colors.RESET}")
                    _flush_text_buffer()
                    print()
                elif text_pending:
                    # 完全没有思考 - 只打印助手头部和文本
                    print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}🤖 助手:{Colors.RESET}")
                    while text_pending:
                        pending = text_pending.pop(0)
                        self._stream_buffer_text.append(pending)
                    _flush_text_buffer()
                    print()

            except Exception as e:
                # 使用结构化错误处理
                llm_error = LLMErrorClassifier.classify(e)

                if llm_error.is_retryable and llm_error.retry_after:
                    print(
                        f"\n{Colors.BRIGHT_YELLOW}请求受限，"
                        f" 等待 {llm_error.retry_after}s 后返回...{Colors.RESET}"
                    )

                error_msg = format_llm_error(e)
                print(f"\n{error_msg}")
                # 自愈：记录LLM错误异常
                self._self_healing.record_anomaly(
                    "llm_error_pattern", 0.7,
                    {"error": str(e)[:200], "step": step}, step,
                )
                self._last_result = f"LLM 调用失败: {llm_error.user_guidance}"
                return self._last_result

            step_runner.process_response(response, step)

            health_issues = step_runner.check_health(step)
            if health_issues:
                for issue in health_issues:
                    print(f"  {Colors.YELLOW}⚠️  {issue}{Colors.RESET}")
                # 自愈：记录健康异常
                self._self_healing.record_anomaly(
                    "health_issues", min(1.0, 0.3 * len(health_issues)),
                    {"count": len(health_issues), "step": step}, step,
                )

            step_runner.prune_thinking()

            # 早期检测循环模式 — 重复后停止
            if step > 2 and step_runner.detect_loop(response):
                self._loop_detection_streak += 1
                self._context.api_total_tokens = self._token_tracker.estimate_tokens(self.messages)
                if self._loop_detection_streak >= 2:
                    loop_msg = response.content or "任务已停止：检测到重复的分析模式。"
                    self._last_result = loop_msg
                    # 自愈：记录循环检测异常
                    self._self_healing.record_anomaly(
                        "loop_detection", 0.85,
                        {"msg": loop_msg[:100], "step": step}, step,
                    )
                    if task_mgr.current_task:
                        task_mgr.current_task.analysis_complete = True
                        task_mgr.end_task()
                    # E3修复：记录循环检测以便调试
                    logger.warning("检测到循环模式两次，正在停止 agent")
                    print(f"\n{Colors.BRIGHT_YELLOW}⚠️  停止运行：检测到重复两次的循环模式{Colors.RESET}")
                    return loop_msg
            else:
                self._loop_detection_streak = 0

            if step_runner.is_complete(response):
                if task_mgr.current_task:
                    task_mgr.end_task()
                step_runner.print_completion_summary(step, step_start_time)
                self._last_result = response.content
                # D11修复：在完成时提取语义记忆
                try:
                    memories = self._semantic_memory.extract_from_session(
                        self.messages, f"completed_step_{step}"
                    )
                    if memories:
                        self._semantic_memory.add_entries(memories)
                except Exception:
                    pass
                self._session_manager.save(
                    self.messages,
                    f"completed_step_{step}",
                    result=self._last_result,
                    analysis=self._last_analysis,
                    state=self._get_runtime_state(),
                )
                return self._last_result

            assert response.tool_calls is not None
            parallel_enabled = self.is_m27 and self.m27_config.get("enable_parallel_tool_calls", True)
            max_concurrent = self.m27_config.get("max_concurrent_tools", 5) if self.is_m27 else 1

            results = await self._execution_engine.execute_tools(
                response.tool_calls,
                max_concurrent,
                parallel_enabled,
                self.mode,
                self._check_approved,
            )

            if len(results) < len(response.tool_calls):
                optimized_calls = [tc for tc, _ in results]
                assistant_msg = Message(
                    role="assistant",
                    content=response.content,
                    thinking=response.thinking,
                    tool_calls=optimized_calls,
                )
                self._context.replace_last_message(assistant_msg)

            # 追加工具消息并处理取消
            for _tool_call, tool_msg in results:
                if self._check_cancelled():
                    self._cleanup_incomplete_messages()
                    self._last_result = "Task cancelled by user."
                    return self._last_result
                self._context.add_message(tool_msg)

            step_elapsed = perf_counter() - step_start_time
            self._metrics.record_step_duration(step_elapsed)

            step_runner.print_step_timing(step, step_start_time)
            step_runner.auto_save(step)

            step += 1

        # 达到最大步数
        error_msg = f"任务在 {self.max_steps} 步后仍未完成。"
        print(f"\n{Colors.BRIGHT_YELLOW}⚠️  {error_msg}{Colors.RESET}")
        # 在达到最大步数时自动保存（用于可能的恢复）
        if self.auto_save:
            try:
                sid = self._session_manager.save(self.messages, f"max_steps_{self.max_steps}", state=self._get_runtime_state())
                print(f"  {Colors.DIM}💾 会话已自动保存: {sid}{Colors.RESET}")
            except Exception as e:
                print(f"  {Colors.DIM}⚠️  自动保存失败: {e}{Colors.RESET}")
        self._last_result = error_msg
        return error_msg

    async def execute_single_tool(self, tool_call: ToolCall) -> tuple[ToolCall, Message]:
        """使用Agent特定行为执行单个工具（打印、记录、批准）。"""
        return await self._execution_engine._execute_single_tool(tool_call, self.mode, self._check_approved)

    async def execute_tools_sequential(self, tool_calls: list[ToolCall]) -> list[tuple[ToolCall, Message]]:
        """逐一执行工具。"""
        return await self._execution_engine._execute_sequential(tool_calls, self.mode, self._check_approved)

    async def execute_tools_parallel(
        self, tool_calls: list[ToolCall], max_concurrent: int = 5
    ) -> list[tuple[ToolCall, Message]]:
        """使用信号量限制并发量并行执行工具。"""
        return await self._execution_engine._execute_parallel(
            tool_calls, max_concurrent, self.mode, self._check_approved
        )

    def _check_approved(self, function_name: str) -> bool:
        """在Agent模式下提示用户批准工具调用。

        如果批准则返回True，如果拒绝则返回False。
        """
        return self._approval_manager.is_approved(function_name)

    def set_mode(self, mode: AgentMode) -> None:
        """切换agent模式。"""
        old_mode = self.mode
        self.mode = mode
        self._context.mode = mode  # 同步到上下文以便健康检查可见
        self._approval_manager.mode = mode
        print(f"{Colors.GREEN}✅ 模式已切换: {old_mode.value} → {mode.value}{Colors.RESET}")

    async def _trigger_self_healing(self, step: int, reason: str) -> None:
        """触发自我修复诊断和可选的自动修复。

        当异常分数超过阈值时调用。对源代码
        运行诊断，并在启用自动修复时可选地应用修复。

        Args:
            step: 当前agent步骤。
            reason: 为什么触发修复。
        """
        status = self._self_healing.get_status()
        top_cats = self._self_healing.get_top_anomaly_categories(3)

        print(
            f"\n{Colors.BRIGHT_CYAN}🩺 触发自我修复 (步骤 {step}, {reason}){Colors.RESET}"
        )
        print(f"  {Colors.DIM}主要异常类型: {', '.join(f'{c}({s:.2f})' for c, s in top_cats)}{Colors.RESET}")

        if not status["auto_heal_enabled"]:
            print(
                f"  {Colors.DIM}自动修复已禁用，仅报告问题。"
                f" 设置 MINI_AGENT_AUTO_HEAL=1 可启用自动修复。{Colors.RESET}"
            )
            print(self._self_healing.get_healing_report())
            return

        try:
            diagnosis = await self._self_healing.diagnose(top_cats)
            fixes = diagnosis.get("suggested_fixes", [])

            if not fixes:
                print(f"  {Colors.DIM}未发现可执行的修复方案。{Colors.RESET}")
                return

            print(f"  {Colors.BRIGHT_YELLOW}发现 {len(fixes)} 个潜在修复方案:{Colors.RESET}")
            for i, fix in enumerate(fixes):
                desc = fix.get("description", fix.get("file", "unknown"))[:80]
                print(f"  {Colors.DIM}  [{i + 1}] {desc}{Colors.RESET}")

            # 应用修复（在非YOLO模式下带批准检查）
            applied = 0
            anomaly_ids = [a.id for a in self._self_healing._anomalies[-5:]]

            for fix in fixes:
                file_name = fix.get("file", "")
                description = fix.get("description", "自动修复")
                old_str = fix.get("old_str", "")
                new_str = fix.get("new_str", "")

                if not file_name or not old_str:
                    continue

                # Agent模式下的批准门控
                if self.mode == AgentMode.AGENT:
                    print(
                        f"  {Colors.BRIGHT_YELLOW}应用修复到 {file_name}？"
                        f" {Colors.DIM}({description[:60]}){Colors.RESET}"
                    )
                    if not self._check_approved(f"heal:{file_name}"):
                        print(f"  {Colors.DIM}  已跳过（用户未批准）{Colors.RESET}")
                        continue

                result = self._self_healing.apply_fix(
                    file_name=file_name,
                    description=description,
                    old_str=old_str,
                    new_str=new_str,
                    anomaly_ids=anomaly_ids,
                )

                if result:
                    applied += 1
                    print(
                        f"  {Colors.BRIGHT_GREEN}✅ 已修复: {file_name} - {description[:60]}"
                        f"{Colors.RESET}"
                    )
                    print(f"  {Colors.DIM}  备份文件: {Path(result.backup_path).name}{Colors.RESET}")

            if applied > 0:
                print(
                    f"  {Colors.BRIGHT_GREEN}🩺 已应用 {applied} 个修复方案。"
                    f" 更改将在下次重启后生效。{Colors.RESET}"
                )
            else:
                print(f"  {Colors.DIM}未应用任何修复。{Colors.RESET}")

        except Exception as e:
            logger.warning("自我修复失败: %s", e)
            print(f"  {Colors.YELLOW}⚠️  自我修复错误: {e}{Colors.RESET}")

    def save_session(self, label: str = "") -> str:
        """保存当前会话，包括最后结果和分析。返回会话ID。"""
        # D11修复：保存前提取语义记忆
        try:
            memories = self._semantic_memory.extract_from_session(self.messages, label or "manual_save")
            if memories:
                added = self._semantic_memory.add_entries(memories)
                if added > 0:
                    print(f"  {Colors.DIM}🧠 已提取 {added} 条语义记忆{Colors.RESET}")
        except Exception:
            pass
        return self._session_manager.save(self.messages, label=label, result=self._last_result, state=self._get_runtime_state())

    def load_session(self, session_id: str) -> bool:
        """加载保存的会话，包括结果、分析和运行时状态。成功时返回True。"""
        messages, result, state = self._session_manager.load(session_id)
        if messages is None:
            return False
        self.replace_messages(messages)
        self._last_result = result  # 恢复最后结果
        # 从元数据中恢复最后分析（如果可用）
        self._last_analysis = self._session_manager.load_analysis(session_id)
        # D3修复：恢复完整的运行时状态（思考预算、循环计数器等）
        self._restore_runtime_state(state)
        # Sync AgentContext state after session restore
        self._sync_context_state()
        return True

    def get_last_result(self) -> str | None:
        """获取上次运行的结果。"""
        return self._last_result

    def set_analysis_result(self, analysis: str) -> None:
        """设置分析结果以便会话持久化。

        这允许agent跨会话记住分析结果。
        在完成分析任务后调用此方法。

        D5修复：分析现在立即持久化到最新的会话文件
        ，防止崩溃时数据丢失。
        """
        self._last_analysis = analysis
        # D5：自动持久化到最新的会话
        try:
            latest_sid = self._session_manager.get_latest_session_id()
            if latest_sid:
                self._session_manager.save_analysis(latest_sid, analysis)
        except Exception:
            pass  # 非关键，不要阻塞agent

    def get_analysis_result(self) -> str | None:
        """获取上次分析结果。"""
        return self._last_analysis

    def _get_runtime_state(self) -> dict[str, Any]:
        """D3修复：为会话持久化序列化所有关键运行时状态。

        捕获在会话恢复时会丢失的状态：
        api_total_tokens、thinking_budget、loop_detection_streak等。
        """
        return {
            "api_call_count": self._context.api_call_count,
            "api_total_tokens": self._context.api_total_tokens,
            "thinking_budget": self.thinking_budget,
            "loop_detection_streak": self._loop_detection_streak,
            "consecutive_failures": self._context.consecutive_failures,
            "last_auto_save_step": self._context.last_auto_save_step,
            "mode": self.mode.value if hasattr(self.mode, "value") else str(self.mode),
        }

    def _restore_runtime_state(self, state: dict[str, Any]) -> None:
        """D3修复：从保存的会话恢复运行时状态。"""
        if not state:
            return
        self._context.api_call_count = state.get("api_call_count", self._context.api_call_count)
        self._context.api_total_tokens = state.get("api_total_tokens", self._context.api_total_tokens)
        self.thinking_budget = state.get("thinking_budget", self.thinking_budget)
        self._loop_detection_streak = state.get("loop_detection_streak", 0)
        self._context.consecutive_failures = state.get("consecutive_failures", 0)
        self._context.last_auto_save_step = state.get("last_auto_save_step", 0)
        print(
            f"{Colors.DIM}  状态已恢复: {state.get('api_call_count', 0)} 次 API 调用，"
            f" {state.get('api_total_tokens', 0)} tokens，"
            f" 预算={state.get('thinking_budget', 'N/A')}{Colors.RESET}"
        )

    def _sync_context_state(self) -> None:
        """在会话恢复后同步AgentContext状态。

        从消息重建token计数和其他派生状态
        以确保会话恢复后的一致性。
        """
        # 从恢复的消息估算总tokens
        estimated_tokens = self._context.estimate_tokens()
        self._context.api_total_tokens = estimated_tokens

        # 重置API调用计数以匹配恢复的消息
        api_calls = sum(1 for m in self._context.get_messages() if m.role == "assistant")
        self._context.api_call_count = api_calls

        # 使token跟踪器缓存失效以强制重新计算
        self._token_tracker.invalidate_cache()

        # 记录恢复的会话信息
        self.logger.debug(f"会话已恢复: {len(self._context.get_messages())} 条消息，约 {estimated_tokens} tokens")

    def list_sessions(self) -> list[dict[str, Any]]:
        """列出所有保存的会话。"""
        return self._session_manager.list_sessions()

    def get_history(self) -> list[Message]:
        """获取消息历史。"""
        return self.messages.copy()

    def get_status(self) -> dict[str, Any]:
        """获取用于自我诊断的agent状态报告。"""
        return self._health_checker.get_status()

    def get_status_report(self) -> str:
        """生成人类可读的状态报告。"""
        return self._health_checker.get_status_report()

    def _check_health(self) -> list[str]:
        """每步后进行自我健康检查。返回发现的问题列表。"""
        return self._health_checker.check().issues

    def get_error_patterns(self) -> dict[str, Any]:
        """获取用于调试和学习的错误模式分析。

        Returns:
            按工具分类的错误模式字典和最近错误历史
        """
        return self._error_recovery.get_patterns()

    def get_suggestions(self) -> list[str]:
        """基于当前agent状态获取建议。

        分析agent状态并提供可操作的建议。
        """
        return self._error_recovery.get_suggestions()

    def get_performance_metrics(self) -> dict[str, Any]:
        """获取当前会话的性能指标。

        Returns:
            包含步骤、工具和API调用时序指标的字典
        """
        return self._metrics.get_metrics()

    async def dispatch_sub_agents(
        self,
        tasks: list[str],
        max_concurrent: int = 3,
        _system_prompt: str = "You are a helpful assistant. Complete the assigned task concisely.",  # noqa: ARG002
    ) -> list[SubAgentResult]:
        """派发多个子agent并行处理独立任务。

        这使agent能够"克隆自己"并同时处理多个问题，
        然后综合结果。

        Args:
            tasks: 子agent的任务描述列表
            max_concurrent: 最大并发子agent数
            system_prompt: 子agent的自定义系统提示词

        Returns:
            包含任务、内容、成功状态、耗时、错误的SubAgentResult对象列表
        """
        from .subagent import run_sub_agents as run_subs

        # 获取一份工具副本给子agent
        tools = self.tool_list

        # 并行运行子agent
        results = await run_subs(
            llm_client=self.llm,
            tasks=tasks,
            tools=tools,
            max_concurrent=max_concurrent,
        )

        # 记录调度信息
        print(
            f"\n{Colors.BRIGHT_CYAN}🔄 已派发 {len(tasks)} 个子 Agent（{max_concurrent} 并发）{Colors.RESET}"
        )
        successful = sum(1 for r in results if r.success)
        print(f"{Colors.BRIGHT_GREEN}✅ 成功: {successful}/{len(tasks)}{Colors.RESET}")

        return results

    def cleanup(self) -> None:
        """清理agent持有的资源。

        当不再需要agent时应调用，以确保
        后台进程和连接的正确清理。
        """
        # 刷新日志记录器
        if hasattr(self, "logger"):
            self.logger.flush()

        # 清除取消事件
        self.cancel_event = None

        # 重置token缓存
        self._token_tracker.invalidate_cache()

        # 注意：不要在这里清除_encoder_cache — 它是模块级
        # 共享缓存。清除它会影响所有Agent实例。

        # 清理后台shell
        from .tools.bash_background import BackgroundShellManager

        try:
            loop = asyncio.get_running_loop()
            # 创建清理任务，在同一事件循环中异步执行
            # 使用create_task调度清理而不阻塞
            loop.create_task(BackgroundShellManager.cleanup_all())
            # 不要await - 让它在后台运行，我们不能在这里阻塞
            # 即使我们返回，任务也会完成
        except RuntimeError:
            logger.warning("清理过程中无运行中的事件循环 - 清理可能被跳过")
        except Exception as e:
            logger.warning("事件循环清理失败: %s", e)

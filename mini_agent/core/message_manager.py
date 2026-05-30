"""核心消息处理与摘要逻辑。"""

from __future__ import annotations

import logging
from typing import Any

from ..schema import Message
from ..utils import Colors
from ..utils.summary_manager import AdaptiveSummaryManager
from .token_tracker import TokenTracker

_logger = logging.getLogger(__name__)


class MessageManager:
    """管理消息历史和摘要。

    处理消息历史、token估算，并在接近token限制时自动进行摘要。
    """

    def __init__(self, token_limit: int):
        self.token_limit = token_limit
        self.messages: list[Message] = []
        self._token_tracker = TokenTracker()
        self._summary_manager = AdaptiveSummaryManager(token_limit)
        self._skip_next_token_check = False
        self._last_summary_quality = 1.0

    def initialize(self, system_prompt: str) -> None:
        """使用系统提示词初始化。"""
        self.messages = [Message(role="system", content=system_prompt)]

    def add_message(self, message: Message) -> None:
        """向历史记录添加消息。"""
        self.messages.append(message)

    def estimate_tokens(self) -> int:
        """估算消息历史的总token数。"""
        return self._token_tracker.estimate_tokens(self.messages)

    def should_summarize(self, api_total_tokens: int) -> tuple[bool, str]:
        """检查是否需要进行摘要。

        返回:
            (是否需要摘要, 原因) 元组
        """
        if self._skip_next_token_check:
            self._skip_next_token_check = False
            return False, ""

        estimated_tokens = self.estimate_tokens()
        should_summarize, reason = self._summary_manager.should_summarize(
            self.messages, estimated_tokens, api_total_tokens
        )
        return should_summarize, reason

    def mark_skip_next_check(self, quality: float = 1.0) -> None:
        """标记跳过下次token检查（在摘要之后）。"""
        self._skip_next_token_check = True
        if self._summary_manager.should_skip_next_check(quality):
            self._skip_next_token_check = True

    def get_messages(self) -> list[Message]:
        """获取当前消息列表。"""
        return self.messages

    def replace_messages(self, messages: list[Message]) -> None:
        """替换整个消息列表。"""
        self.messages = messages

    async def summarize_messages(
        self,
        messages: list[Message],
        api_total_tokens: int,
        logger: Any,
        _max_truncation: int = 2000,
    ) -> list[Message]:
        """当token超出限制时对消息历史进行摘要。

        策略:
        - 保留所有用户消息（这些代表用户意图）
        - 对每对用户-用户之间的内容进行摘要（智能体执行过程）
        - 如果最后一轮仍在执行中，同样进行摘要
        - 结构: system -> user1 -> summary1 -> user2 -> summary2 -> ...

        参数:
            messages: 当前消息列表（将在原地修改）
            api_total_tokens: 上次API响应报告的token计数
            logger: 用于警告的日志记录器实例
            max_truncation: 摘要内容的最大截断长度

        返回:
            摘要后的新消息列表（如果不需要摘要则返回相同列表）
        """
        if self._skip_next_token_check:
            self._skip_next_token_check = False
            return messages

        estimated_tokens = self._token_tracker.estimate_tokens(messages)

        should_summarize, reason = self._summary_manager.should_summarize(messages, estimated_tokens, api_total_tokens)

        if not should_summarize:
            return messages

        print(
            f"\n{Colors.BRIGHT_YELLOW}📊 Token usage - Local estimate: {estimated_tokens},"
            f" API reported: {api_total_tokens}, Limit: {self.token_limit}{Colors.RESET}"
        )
        print(f"{Colors.BRIGHT_YELLOW}🔄 Triggering message summarization ({reason})...{Colors.RESET}")

        user_indices = [i for i, msg in enumerate(messages) if msg.role == "user" and i > 0]

        if len(user_indices) < 1:
            print(f"{Colors.BRIGHT_YELLOW}⚠️  Insufficient messages, cannot summarize{Colors.RESET}")
            return messages

        # D12 修复: 检测摘要生成深度，防止级联退化。
        # 每次对摘要再次进行摘要时，需要更高的保留度。
        summary_generation = self._detect_summary_generation(messages)
        if summary_generation > 0:
            print(
                f"{Colors.BRIGHT_YELLOW}🔍 Summary generation {summary_generation} detected"
                f" - applying anti-degeneration boost{Colors.RESET}"
            )

        new_messages = [messages[0]]
        summary_count = 0

        # D12 修复: 反退化最小保留比例
        # 世代 0（全新）: 正常比例
        # 世代 1: +20% 提升
        # 世代 2+: +35% 提升并保留关键事实
        gen_boost = min(0.35, summary_generation * 0.15)

        for i, user_idx in enumerate(user_indices):
            new_messages.append(messages[user_idx])

            user_content = messages[user_idx].content
            if isinstance(user_content, str) and len(user_content) > 5000:
                logger.warning(
                    "User message truncated from %d to 5000 chars during summarization",
                    len(user_content),
                )
                messages[user_idx].content = user_content[:5000] + "...[truncated]..."

            next_user_idx = user_indices[i + 1] if i < len(user_indices) - 1 else len(messages)

            execution_messages = messages[user_idx + 1 : next_user_idx]

            if execution_messages:
                is_last_round = (i == len(user_indices) - 1)

                # 关键修复: 对于最近的轮次，原样保留助手消息而非进行摘要。
                # 因为用户的后续指令几乎总是引用AI最近的输出。对其进行摘要会导致
                # "AI忘记自己刚说过的话"的问题。
                if is_last_round and self._should_preserve_last_round(execution_messages):
                    for msg in execution_messages:
                        if msg.role in ("assistant", "tool"):
                            new_messages.append(msg)
                    summary_count += 1  # 算作"已处理"，尽管是保留而非摘要
                    continue

                tier = "medium"
                if ":" in reason and "tier=" in reason:
                    # D12 修复: 从原因中解析tier，处理 degen_w 后缀
                    tier_part = reason.split("tier=")[1]
                    tier = tier_part.split(":")[0]  # 去掉 :degen_w{N} 后缀
                elif reason.startswith("early_trigger"):
                    tier = "low"

                summary_config = self._summary_manager.get_summary_config(tier)
                # D12 修复: 应用反退化提升
                # 世代越高 → 保留度越高，防止级联丢失
                boosted_ratio = min(1.0, summary_config["preserve_ratio"] + gen_boost)
                boosted_truncation = summary_config["max_truncation"]
                if summary_generation > 0:
                    # 重新摘要时不进行过度截断
                    boosted_truncation = min(summary_config["max_truncation"] * 2, 3000)
                summary_text = self._create_local_summary(
                    execution_messages,
                    i + 1,
                    preserve_ratio=boosted_ratio,
                    max_truncation=boosted_truncation,
                )
                if summary_text:
                    summary_message = Message(
                        role="user",
                        content=f"[Execution Summary {i + 1}]\n\n{summary_text}",
                    )
                    new_messages.append(summary_message)
                    summary_count += 1

        self._skip_next_token_check = True

        self._last_summary_quality = self._summary_manager.estimate_summary_quality(
            new_messages,
            str(len(new_messages) * 100),
        )

        if self._summary_manager.should_skip_next_check(self._last_summary_quality):
            self._skip_next_token_check = True

        new_tokens = self._token_tracker.estimate_tokens(new_messages)

        print(
            f"{Colors.BRIGHT_GREEN}✓ Summary completed, local tokens: {estimated_tokens} → {new_tokens}{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}  Structure: system + {len(user_indices)} user messages"
            f" + {summary_count} summaries{Colors.RESET}"
        )
        print(
            f"{Colors.DIM}  Quality: {self._last_summary_quality:.2f} |"
            f" Note: API token count will update on next LLM call{Colors.RESET}"
        )

        return new_messages

    @staticmethod
    def _detect_summary_generation(messages: list[Message]) -> int:
        """D12 修复: 检测已发生摘要的代数。

        扫描用户消息中的 [Execution Summary] 标记以确定
        摘要链的深度。每增加一层摘要
        就增加一代。

        返回:
            摘要世代深度（0 = 全新，1+ = 重新摘要）。
        """
        max_gen = 0
        for msg in messages:
            if msg.role == "user" and msg.content and isinstance(msg.content, str):
                content = msg.content
                # Count nested summary markers
                if "[Execution Summary" in content:
                    gen = content.count("[Execution Summary")
                    max_gen = max(max_gen, gen)
                if "Stats:" in content and "tool(s)" in content:
                    # Also count re-summarized stats
                    gen = content.count("tool(s)")
                    max_gen = max(max_gen, gen)
        return max_gen

    @staticmethod
    def _should_preserve_last_round(execution_messages: list[Message]) -> bool:
        """判断是否应原样保留上一轮助手的消息。

        当AI刚刚提供了用户下一条消息会引用的分析/建议时，
        对这些消息进行摘要会导致AI"忘记"自己刚说的话。

        如果任何助手消息有实质性文本内容（超过100个字符），
        我们就保留最后一轮——这表明是有意义的输出，
        用户可能会跟进。
        """
        for msg in execution_messages:
            if msg.role == "assistant":
                content = msg.content if isinstance(msg.content, str) else str(msg.content) if msg.content else ""
                if len(content) > 100:
                    return True
        return False

    def _create_local_summary(
        self,
        messages: list[Message],
        round_num: int,
        preserve_ratio: float = 0.6,
        max_truncation: int = 1000,
    ) -> str:
        """在本地创建摘要而不调用LLM（节省token）。

        保留更多细节以减少冗余的LLM调用。

        参数:
            messages: 要摘要的消息列表
            round_num: 轮次编号
            preserve_ratio: 保留多少内容 (0-1)
            max_truncation: 最大截断字符数

        返回:
            摘要文本
        """
        if not messages:
            return ""

        lines = [f"Round {round_num}:"]
        tool_calls_count = 0
        tool_results_count = 0
        assistant_responses = []

        for msg in messages:
            if msg.role == "assistant":
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content:
                    if len(content) > max_truncation:
                        content = content[:max_truncation] + "..."
                    assistant_responses.append(content)
                if msg.tool_calls:
                    tool_details = []
                    for tc in msg.tool_calls:
                        args_str = str(tc.function.arguments)
                        args_max = max(150, int(150 * preserve_ratio * 1.5))
                        if len(args_str) > args_max:
                            args_str = args_str[:args_max] + "..."
                        tool_details.append(f"{tc.function.name}({args_str})")
                    lines.append(f"  Tools called: {', '.join(tool_details)}")
                    tool_calls_count += len(msg.tool_calls)
            elif msg.role == "tool":
                tool_results_count += 1
                result = msg.content if isinstance(msg.content, str) else str(msg.content)
                result_max = max(1500, int(1500 * preserve_ratio * 1.5))
                if len(result) > result_max:
                    result = result[:result_max] + "..."
                lines.append(f"  Result: {result}")

        # 关键修复: 在摘要中包含助手的分析回复
        # 之前，assistant_responses 被收集，但仅在没有 tool_calls 时输出。
        # 在分析场景中，AI读取文件（tool_calls）同时给出建议（assistant content）。
        # 建议被静默丢弃，导致AI"忘记"自己刚说的话
        # 并重复分析相同的文件。
        if assistant_responses:
            for resp in assistant_responses:
                # Truncate long assistant responses proportionally
                resp_max = max(2000, int(2000 * preserve_ratio * 1.5))
                if len(resp) > resp_max:
                    resp = resp[:resp_max] + "..."
                lines.append(f"  Assistant: {resp}")

        if tool_calls_count or tool_results_count:
            lines.append(f"  Stats: {tool_calls_count} tool(s), {tool_results_count} result(s)")

        return "\n".join(lines)

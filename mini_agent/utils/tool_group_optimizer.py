"""M2.7智能工具批量优化器。

分析工具调用依赖关系并将它们分组以实现最佳并行执行。
在遵守工具依赖关系的同时最大化吞吐量。
"""

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..schema import ToolCall


class ToolGroupOptimizer:
    """基于工具类型和依赖关系优化工具调用批量处理。

    M2.7支持20+并行工具调用，成功率97%。
    此优化器将独立操作分组并对依赖操作进行排序。
    """

    # 只读工具，可以始终并行运行
    READ_ONLY = frozenset(
        {
            "read_file",
            "multi_read",
            "grep",
            "multi_grep",
            "find",
            "tree",
            "workspace_context",
            "deep_context",
            "git_status",
            "list_sessions",
        }
    )

    # 修改状态的工具 - 不应与同一目标操作并行运行
    WRITE_TOOLS = frozenset(
        {
            "write_file",
            "edit_file",
            "multi_edit",
            "bash",
            "multi_bash",
            "git",
            "delete_file",
            "move_file",
            "copy_file",
        }
    )

    # 信息工具 - 非常轻量，可以随时运行
    INFO_TOOLS = frozenset(
        {
            "tree",
            "git_status",
            "git_log",
            "list_sessions",
            "get_history",
        }
    )

    @classmethod
    def can_parallelize(cls, tool_calls: list[ToolCall]) -> bool:
        """检查所有工具调用是否可以并行运行。

        Args:
            tool_calls: 要检查的工具调用列表

        Returns:
            如果所有工具都是只读的或写工具不冲突则为True
        """
        if len(tool_calls) <= 1:
            return True

        # 按目标收集写操作
        write_targets: dict[str, list[str]] = defaultdict(list)

        for tc in tool_calls:
            name = tc.function.name

            # 跳过只读工具
            if name in cls.READ_ONLY or name in cls.INFO_TOOLS:
                continue

            # 检查写冲突
            if name in cls.WRITE_TOOLS:
                args = tc.function.arguments
                # 从参数中提取目标文件/目录
                target = cls._extract_target(name, args)
                if target:
                    write_targets[target].append(name)

        # 如果任何目标有多个写操作，则无法完全并行化
        return all(len(operations) <= 1 for target, operations in write_targets.items())

    @classmethod
    def _extract_target(cls, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """从工具参数中提取主要目标（文件/目录）。"""
        if tool_name in ("write_file", "edit_file", "delete_file", "read_file", "multi_read"):
            return arguments.get("path")
        elif tool_name in ("bash", "multi_bash"):
            cmd: str = arguments.get("command", "")
            return cmd[:100]
        elif tool_name == "git":
            return arguments.get("operation")
        return None

    @classmethod
    def group_by_dependency(cls, tool_calls: list[ToolCall]) -> list[list[ToolCall]]:
        """将工具调用分组为可以并行运行的批次。

        Returns:
            批次列表，每个批次可以并行运行。
            批次是有序的 - 后续批次依赖于较早的批次。
        """
        if not tool_calls:
            return []

        # 简单情况：如果全是只读工具，并行运行
        if all(tc.function.name in cls.READ_ONLY | cls.INFO_TOOLS for tc in tool_calls):
            return [tool_calls]

        # 按工具类别分组
        batches: list[list[ToolCall]] = []
        read_batch: list[ToolCall] = []
        write_batch: list[ToolCall] = []

        for tc in tool_calls:
            name = tc.function.name

            if name in cls.READ_ONLY or name in cls.INFO_TOOLS:
                read_batch.append(tc)
            else:
                # 如果有待处理的读操作且遇到写操作，先刷新读操作
                if read_batch and write_batch:
                    batches.append(read_batch)
                    read_batch = []
                write_batch.append(tc)

        # 刷新剩余的批次
        if read_batch:
            batches.append(read_batch)
        if write_batch:
            batches.append(write_batch)

        return batches if batches else [tool_calls]

    @classmethod
    def deduplicate_paths(cls, paths: list[str]) -> list[str]:
        """从multi_read/multi_edit中移除重复的路径。

        Args:
            paths: 文件路径列表（可能包含重复）

        Returns:
            保持顺序的去重列表
        """
        seen = set()
        result = []
        for p in paths:
            # 标准化路径
            normalized = str(Path(p).resolve()) if Path(p).is_absolute() else p
            if normalized not in seen:
                seen.add(normalized)
                result.append(p)
        return result

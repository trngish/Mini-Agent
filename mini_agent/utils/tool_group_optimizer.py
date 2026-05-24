"""Intelligent tool batch optimizer for M2.7.

Analyzes tool call dependencies and groups them for optimal parallel execution.
Maximizes throughput while respecting tool dependencies.
"""

from collections import defaultdict
from typing import Any

from ..schema import ToolCall


class ToolGroupOptimizer:
    """Optimizes tool call batching based on tool types and dependencies.
    
    M2.7 supports 20+ parallel tool calls with 97% following rate.
    This optimizer groups independent operations and sequences dependent ones.
    """

    # Tools that are read-only and can always run in parallel
    READ_ONLY = frozenset({
        'read_file', 'multi_read', 'grep', 'multi_grep', 'find', 'tree',
        'workspace_context', 'deep_context', 'git_status', 'list_sessions',
    })

    # Tools that modify state - should not run in parallel with same-target operations
    WRITE_TOOLS = frozenset({
        'write_file', 'edit_file', 'multi_edit', 'bash', 'multi_bash', 'git',
        'delete_file', 'move_file', 'copy_file',
    })

    # Info tools - very cheap, can run anytime
    INFO_TOOLS = frozenset({
        'tree', 'git_status', 'git_log', 'list_sessions', 'get_history',
    })

    @classmethod
    def can_parallelize(cls, tool_calls: list[ToolCall]) -> bool:
        """Check if all tool calls can run in parallel.
        
        Args:
            tool_calls: List of tool calls to check
            
        Returns:
            True if all tools are read-only or write tools don't conflict
        """
        if len(tool_calls) <= 1:
            return True

        # Collect write operations by target
        write_targets: dict[str, list[str]] = defaultdict(list)
        
        for tc in tool_calls:
            name = tc.function.name
            
            # Skip read-only tools
            if name in cls.READ_ONLY or name in cls.INFO_TOOLS:
                continue
            
            # Check for write conflicts
            if name in cls.WRITE_TOOLS:
                args = tc.function.arguments
                # Extract target file/dir from arguments
                target = cls._extract_target(name, args)
                if target:
                    write_targets[target].append(name)

        # If any target has multiple write operations, can't fully parallelize
        for target, operations in write_targets.items():
            if len(operations) > 1:
                return False

        return True

    @classmethod
    def _extract_target(cls, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """Extract the primary target (file/directory) from tool arguments."""
        if tool_name in ('write_file', 'edit_file', 'delete_file', 'read_file', 'multi_read'):
            return arguments.get('path')
        elif tool_name in ('bash', 'multi_bash'):
            return arguments.get('command', '')[:100]  # Use command prefix as target
        elif tool_name == 'git':
            return arguments.get('operation')
        return None

    @classmethod
    def group_by_dependency(cls, tool_calls: list[ToolCall]) -> list[list[ToolCall]]:
        """Group tool calls into batches that can run in parallel.
        
        Returns:
            List of batches, where each batch can run in parallel.
            Batches are ordered - later batches depend on earlier ones.
        """
        if not tool_calls:
            return []

        # Simple case: if all read-only, run in parallel
        if all(tc.function.name in cls.READ_ONLY | cls.INFO_TOOLS for tc in tool_calls):
            return [tool_calls]

        # Group by tool category
        batches: list[list[ToolCall]] = []
        read_batch: list[ToolCall] = []
        write_batch: list[ToolCall] = []

        for tc in tool_calls:
            name = tc.function.name
            
            if name in cls.READ_ONLY or name in cls.INFO_TOOLS:
                read_batch.append(tc)
            else:
                # If we have pending reads and encounter a write, flush reads first
                if read_batch and write_batch:
                    batches.append(read_batch)
                    read_batch = []
                write_batch.append(tc)

        # Flush remaining batches
        if read_batch:
            batches.append(read_batch)
        if write_batch:
            batches.append(write_batch)

        return batches if batches else [tool_calls]

    @classmethod
    def deduplicate_paths(cls, paths: list[str]) -> list[str]:
        """Remove duplicate paths from multi_read/multi_edit.
        
        Args:
            paths: List of file paths (may contain duplicates)
            
        Returns:
            Deduplicated list preserving order
        """
        seen = set()
        result = []
        for p in paths:
            # Normalize path
            normalized = str(Path(p).resolve()) if Path(p).is_absolute() else p
            if normalized not in seen:
                seen.add(normalized)
                result.append(p)
        return result


# Import Path for _extract_target
from pathlib import Path
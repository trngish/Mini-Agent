"""Task state manager for preventing infinite loops and tracking progress.

Provides a state machine for complex tasks:
1. Tracks which files have been read
2. Tracks completed task phases
3. Determines when task is "done"
4. Prevents redundant operations
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class TaskPhase(Enum):
    """Task execution phases."""

    INITIAL = auto()  # Task just started
    READING = auto()  # Reading files
    ANALYZING = auto()  # Analyzing code
    PLANNING = auto()  # Planning changes
    MODIFYING = auto()  # Modifying code
    VERIFYING = auto()  # Verifying changes
    COMPLETE = auto()  # Task complete
    CANCELLED = auto()  # Task cancelled


@dataclass
class TaskContext:
    """Context for a single task execution."""

    task_id: str
    description: str
    phase: TaskPhase = TaskPhase.INITIAL
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    files_read: set[str] = field(default_factory=set)
    files_modified: set[str] = field(default_factory=set)
    steps_completed: int = 0
    steps_limit: int = 100
    analysis_complete: bool = False
    modifications_complete: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_file_read(self, path: str) -> None:
        """Mark a file as read."""
        self.files_read.add(path)
        self.updated_at = time.time()

    def mark_file_modified(self, path: str) -> None:
        """Mark a file as modified."""
        self.files_modified.add(path)
        self.updated_at = time.time()

    def advance_phase(self, new_phase: TaskPhase) -> None:
        """Advance to a new phase."""
        self.phase = new_phase
        self.updated_at = time.time()

    def increment_steps(self) -> None:
        """Increment step counter."""
        self.steps_completed += 1

    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.phase == TaskPhase.COMPLETE or self.phase == TaskPhase.CANCELLED

    def should_stop(self) -> tuple[bool, str]:
        """Check if task should stop based on conditions.

        Returns:
            Tuple of (should_stop, reason)
        """
        # Check for completion markers in analysis tasks
        if self.phase == TaskPhase.ANALYZING and self.analysis_complete:
            return True, "Analysis complete"

        # Check for completion markers in modification tasks
        if self.phase == TaskPhase.MODIFYING and self.modifications_complete:
            return True, "Modifications complete"

        # Check for stuck state (no progress in 10+ steps)
        if self.phase in (TaskPhase.ANALYZING, TaskPhase.PLANNING) and self.steps_completed > 10 and not self.files_read:
            return True, "No progress detected - possible loop"

        return False, ""


class TaskStateManager:
    """Manages task state across agent execution.

    Singleton pattern to track task state globally.
    """

    _instance: TaskStateManager | None = None

    def __new__(cls) -> TaskStateManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._current_task: TaskContext | None = None
            cls._instance._task_history: list[TaskContext] = []
        return cls._instance

    @property
    def current_task(self) -> TaskContext | None:
        """Get current task context."""
        return self._current_task

    def start_task(self, task_id: str, description: str, max_steps: int = 100) -> TaskContext:
        """Start a new task."""
        task = TaskContext(
            task_id=task_id,
            description=description,
            steps_limit=max_steps,
        )
        self._current_task = task
        self._task_history.append(task)
        return task

    def get_task(self) -> TaskContext | None:
        """Get current task."""
        return self._current_task

    def end_task(self) -> None:
        """End current task."""
        if self._current_task:
            self._current_task.advance_phase(TaskPhase.COMPLETE)
        self._current_task = None

    def cancel_task(self) -> None:
        """Cancel current task."""
        if self._current_task:
            self._current_task.advance_phase(TaskPhase.CANCELLED)
        self._current_task = None

    def mark_file_read(self, path: str) -> None:
        """Mark file as read in current task."""
        if self._current_task:
            self._current_task.mark_file_read(path)

    def mark_file_modified(self, path: str) -> None:
        """Mark file as modified in current task."""
        if self._current_task:
            self._current_task.mark_file_modified(path)

    def is_file_read(self, path: str) -> bool:
        """Check if file was already read."""
        if self._current_task:
            return path in self._current_task.files_read
        return False

    def is_file_modified(self, path: str) -> bool:
        """Check if file was already modified."""
        if self._current_task:
            return path in self._current_task.files_modified
        return False

    def increment_steps(self) -> None:
        """Increment step counter."""
        if self._current_task:
            self._current_task.increment_steps()

    def check_should_stop(self) -> tuple[bool, str]:
        """Check if current task should stop.

        Returns:
            Tuple of (should_stop, reason)
        """
        if self._current_task:
            return self._current_task.should_stop()
        return False, ""

    def get_status_report(self) -> str:
        """Get human-readable status report."""
        if not self._current_task:
            return "No active task"

        task = self._current_task
        lines = [
            f"Task: {task.task_id}",
            f"Description: {task.description}",
            f"Phase: {task.phase.name}",
            f"Steps: {task.steps_completed}/{task.steps_limit}",
            f"Files read: {len(task.files_read)}",
            f"Files modified: {len(task.files_modified)}",
            f"Duration: {time.time() - task.created_at:.1f}s",
        ]

        if task.analysis_complete:
            lines.append("Analysis: COMPLETE")
        if task.modifications_complete:
            lines.append("Modifications: COMPLETE")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset task state (for testing)."""
        self._current_task = None
        self._task_history.clear()


# Pattern matchers for detecting "done" states in analysis output
COMPLETION_PATTERNS = [
    # English
    re.compile(r"(?i)analysis complete[d]?"),
    re.compile(r"(?i)report complete[d]?"),
    re.compile(r"(?i)optimization complete[d]?"),
    re.compile(r"(?i)all issues addressed"),
    re.compile(r"(?i)## summary", re.M),  # Markdown summary section
    re.compile(r"(?i)^\*{3,}$", re.M),  # Horizontal rule at end
    # Chinese
    re.compile(r"(?i)分析完成"),
    re.compile(r"(?i)报告完成"),
    re.compile(r"(?i)优化完成"),
    re.compile(r"(?i)所有问题已解决"),
]

# Patterns that indicate a loop/stall state
LOOP_PATTERNS = [
    re.compile(r"(?i)let me (read|analyze) .+ again"),
    re.compile(r"(?i)let me (re-)?read"),
    re.compile(r"(?i)based on (my )?(previous|earlier) (read|analysis)"),
    re.compile(r"(?i)from the (context|previous) (read|analysis)"),
    re.compile(r"(?i)earlier (read|analysis|execution)"),
    re.compile(r"(?i)re-?(read|analyze|examine)"),
    # Additional patterns for common looping phrases
    re.compile(r"(?i)let me also (read|analyze|check)"),
    re.compile(r"(?i)i already (read|analyzed|saw)"),
    re.compile(r"(?i)as mentioned (above|earlier)"),
    re.compile(r"(?i)as shown (above|earlier)"),
    re.compile(r"(?i)going back to (the )?(file|code|content)"),
]


def detect_completion(text: str) -> bool:
    """Detect if text indicates task completion."""
    return any(pattern.search(text) for pattern in COMPLETION_PATTERNS)


def detect_loop_pattern(text: str) -> bool:
    """Detect if text shows loop/repetition patterns."""
    return any(pattern.search(text) for pattern in LOOP_PATTERNS)


# Global instance accessor
def get_task_manager() -> TaskStateManager:
    """Get the global TaskStateManager instance."""
    return TaskStateManager()

"""Tests for task state management and loop detection."""

import pytest

from mini_agent.utils.task_state import (
    TaskPhase,
    TaskStateManager,
    detect_completion,
    detect_loop_pattern,
    get_task_manager,
)


class TestTaskStateManager:
    """Tests for TaskStateManager."""

    def setup_method(self):
        """Reset global state before each test."""
        tm = get_task_manager()
        tm.reset()

    def test_start_task(self):
        """Test starting a new task."""
        tm = get_task_manager()
        task = tm.start_task("test-1", "Testing task")

        assert task.task_id == "test-1"
        assert task.description == "Testing task"
        assert task.phase == TaskPhase.INITIAL
        assert tm.current_task is task

    def test_mark_file_read(self):
        """Test marking files as read."""
        tm = get_task_manager()
        tm.start_task("test", "Test")

        tm.mark_file_read("main.py")
        tm.mark_file_read("config.py")

        assert "main.py" in tm.current_task.files_read
        assert "config.py" in tm.current_task.files_read
        assert tm.is_file_read("main.py")
        assert not tm.is_file_read("other.py")

    def test_mark_file_modified(self):
        """Test marking files as modified."""
        tm = get_task_manager()
        tm.start_task("test", "Test")

        tm.mark_file_modified("utils.py")

        assert "utils.py" in tm.current_task.files_modified
        assert tm.is_file_modified("utils.py")

    def test_phase_advancement(self):
        """Test advancing task phases."""
        tm = get_task_manager()
        tm.start_task("test", "Test")

        tm.current_task.advance_phase(TaskPhase.READING)
        assert tm.current_task.phase == TaskPhase.READING

        tm.current_task.advance_phase(TaskPhase.ANALYZING)
        assert tm.current_task.phase == TaskPhase.ANALYZING

    def test_end_task(self):
        """Test ending a task."""
        tm = get_task_manager()
        tm.start_task("test", "Test")

        tm.end_task()
        assert tm.current_task is None
        assert tm._task_history[-1].phase == TaskPhase.COMPLETE

    def test_cancel_task(self):
        """Test cancelling a task."""
        tm = get_task_manager()
        tm.start_task("test", "Test")

        tm.cancel_task()
        assert tm.current_task is None
        assert tm._task_history[-1].phase == TaskPhase.CANCELLED

    def test_should_stop_no_progress(self):
        """Test should_stop returns True when analyzing with no files read."""
        tm = get_task_manager()
        tm.start_task("test", "Test", max_steps=100)

        tm.current_task.advance_phase(TaskPhase.ANALYZING)
        for _ in range(11):
            tm.current_task.increment_steps()

        should_stop, reason = tm.check_should_stop()
        assert should_stop
        assert "No progress" in reason

    def test_should_stop_analysis_complete(self):
        """Test should_stop when analysis is marked complete."""
        tm = get_task_manager()
        tm.start_task("test", "Test")

        tm.current_task.advance_phase(TaskPhase.ANALYZING)
        tm.current_task.analysis_complete = True

        should_stop, reason = tm.check_should_stop()
        assert should_stop
        assert "Analysis complete" in reason

    def test_get_status_report(self):
        """Test status report generation."""
        tm = get_task_manager()
        tm.start_task("my-task", "Test task")
        tm.mark_file_read("a.py")
        tm.mark_file_read("b.py")
        tm.mark_file_modified("c.py")

        report = tm.get_status_report()
        assert "my-task" in report
        assert "Test task" in report
        assert "2" in report  # files read


class TestCompletionDetection:
    """Tests for completion pattern detection."""

    @pytest.mark.parametrize("text,expected", [
        ("Analysis complete", True),
        ("analysis complete", True),
        ("ANALYSIS COMPLETE", True),
        ("分析完成", True),
        ("分析完成了", True),
        ("Report completed", True),
        ("## Summary", True),
        ("optimization complete", True),
        ("all issues addressed", True),
        ("This is some text", False),
        ("No completion here", False),
        ("", False),
    ])
    def test_completion_patterns(self, text, expected):
        """Test various completion patterns."""
        assert detect_completion(text) == expected


class TestLoopDetection:
    """Tests for loop pattern detection."""

    @pytest.mark.parametrize("text,expected", [
        ("Let me read the file again", True),
        ("Let me re-read the file", True),
        ("Based on previous analysis", True),
        ("from the earlier read", True),
        ("re-examine the code", True),
        ("This is new work", False),
        ("I haven't looked at this yet", False),
        ("Let me analyze the code", False),  # "again" is required
        ("", False),
    ])
    def test_loop_patterns(self, text, expected):
        """Test various loop patterns."""
        assert detect_loop_pattern(text) == expected

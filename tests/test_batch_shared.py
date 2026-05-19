"""Tests for batch shared utilities."""

import os
import tempfile
from pathlib import Path

import pytest

from mini_agent.tools.batch_shared import get_git_status_sync, get_tree_sync


class TestGetTreeSync:
    """Tests for get_tree_sync function."""

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_tree_sync(tmpdir, max_depth=3)
            assert "." in result  # Root directory should be listed

    def test_single_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.py").write_text("print('hello')")
            result = get_tree_sync(tmpdir, max_depth=3)
            assert "test.py" in result

    def test_nested_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "src" / "app"
            nested.mkdir(parents=True)
            (nested / "main.py").write_text("print('hello')")
            result = get_tree_sync(tmpdir, max_depth=3)
            assert "src" in result
            assert "app" in result
            assert "main.py" in result

    def test_max_depth_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deep = Path(tmpdir) / "a" / "b" / "c" / "d"
            deep.mkdir(parents=True)
            (deep / "file.txt").write_text("deep")
            result = get_tree_sync(tmpdir, max_depth=2)
            # Should not show files at depth 4
            assert "file.txt" not in result

    def test_skips_hidden_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".hidden").mkdir()
            (Path(tmpdir) / ".hidden" / "secret.txt").write_text("secret")
            (Path(tmpdir) / "visible").mkdir()
            (Path(tmpdir) / "visible" / "public.txt").write_text("public")
            result = get_tree_sync(tmpdir, max_depth=3)
            assert ".hidden" not in result
            assert "visible" in result
            assert "public.txt" in result

    def test_skips_common_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "node_modules").mkdir()
            (Path(tmpdir) / "__pycache__").mkdir()
            (Path(tmpdir) / "src").mkdir()
            result = get_tree_sync(tmpdir, max_depth=3)
            assert "node_modules" not in result
            assert "__pycache__" not in result
            assert "src" in result

    def test_show_sizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "small.txt").write_text("hi")
            result = get_tree_sync(tmpdir, max_depth=3, show_sizes=True)
            assert "small.txt" in result
            # Should contain size info
            assert "B" in result or "K" in result

    def test_max_files_per_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(25):
                (Path(tmpdir) / f"file_{i:02d}.txt").write_text(f"content {i}")
            result = get_tree_sync(tmpdir, max_depth=3, max_files_per_dir=20)
            assert "more files" in result

    def test_unlimited_files_per_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(30):
                (Path(tmpdir) / f"file_{i:02d}.txt").write_text(f"content {i}")
            result = get_tree_sync(tmpdir, max_depth=3, max_files_per_dir=0)
            # All 30 files should be present
            assert "file_29" in result

    def test_nonexistent_dir(self):
        result = get_tree_sync("/nonexistent/path/xyz", max_depth=3)
        # Should handle gracefully
        assert isinstance(result, str)


class TestGetGitStatusSync:
    """Tests for get_git_status_sync function."""

    def test_non_git_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_git_status_sync(tmpdir)
            assert "Not a git repository" in result

    def test_git_repo_status(self):
        """Test with an actual git repo (if git is available)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to init a git repo
            ret = os.system(f'cd "{tmpdir}" && git init > /dev/null 2>&1')
            if ret != 0:
                pytest.skip("git not available")
            result = get_git_status_sync(tmpdir)
            assert "Branch:" in result or "Not a git repository" in result

    def test_max_status_lines_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_git_status_sync(tmpdir, max_status_lines=30, max_commits=5)
            assert isinstance(result, str)

    def test_max_commits_parameter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_git_status_sync(tmpdir, max_commits=10)
            assert isinstance(result, str)

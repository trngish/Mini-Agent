from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from mini_agent.ui import (
    _open_directory_in_file_manager,
    get_log_directory,
    print_banner,
    print_help,
    print_image,
    print_session_info,
    print_stats,
    read_log_file,
    show_log_directory,
)


def _make_mock_agent(mode="agent", messages=None, api_call_count=0, workspace_dir="."):
    agent = MagicMock()
    agent.mode = MagicMock()
    agent.mode.value = mode
    agent.messages = messages if messages is not None else []
    agent.api_call_count = api_call_count
    agent.workspace_dir = Path(workspace_dir)
    return agent


class TestPrintBanner:
    def test_prints_without_error(self, capsys):
        print_banner()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_output_contains_logo(self, capsys):
        print_banner()
        captured = capsys.readouterr()
        assert "╔" in captured.out or "╦" in captured.out


class TestPrintHelp:
    def test_prints_help_commands(self, capsys):
        print_help()
        captured = capsys.readouterr()
        assert "/help" in captured.out
        assert "/clear" in captured.out
        assert "/exit" in captured.out

    def test_prints_key_bindings(self, capsys):
        print_help()
        captured = capsys.readouterr()
        assert "Tab" in captured.out
        assert "Ctrl+C" in captured.out


class TestPrintSessionInfo:
    def test_prints_session_info(self, capsys):
        agent = _make_mock_agent(mode="yolo")
        workspace = Path("/tmp/workspace")
        print_session_info(agent, workspace, "test-model")
        captured = capsys.readouterr()
        assert "test-model" in captured.out
        assert "YOLO" in captured.out

    def test_prints_different_modes(self, capsys):
        agent = _make_mock_agent(mode="plan")
        workspace = Path("/tmp/workspace")
        print_session_info(agent, workspace, "gpt-4")
        captured = capsys.readouterr()
        assert "PLAN" in captured.out


class TestPrintStats:
    def test_prints_stats(self, capsys):
        agent = _make_mock_agent(messages=["a", "b", "c"], api_call_count=5)
        session_start = datetime(2025, 1, 1, 10, 0, 0)
        with patch("mini_agent.ui.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 1, 10, 5, 30)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            print_stats(agent, session_start)
        captured = capsys.readouterr()
        assert "3" in captured.out
        assert "5" in captured.out

    def test_prints_zero_stats(self, capsys):
        agent = _make_mock_agent(messages=[], api_call_count=0)
        session_start = datetime(2025, 1, 1, 10, 0, 0)
        with patch("mini_agent.ui.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2025, 1, 1, 10, 0, 10)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            print_stats(agent, session_start)
        captured = capsys.readouterr()
        assert "0" in captured.out


class TestGetLogDirectory:
    def test_returns_expected_path(self):
        result = get_log_directory()
        assert result == Path.home() / ".mini-agent" / "log"

    def test_returns_path_instance(self):
        result = get_log_directory()
        assert isinstance(result, Path)


class TestShowLogDirectory:
    def test_when_dir_does_not_exist(self, capsys):
        with patch("mini_agent.ui.get_log_directory", return_value=Path("/nonexistent/path")):
            show_log_directory(open_file_manager=False)
        captured = capsys.readouterr()
        assert "does not exist" in captured.out

    def test_when_dir_exists(self, capsys, tmp_path):
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        (log_dir / "test.log").write_text("hello")
        with patch("mini_agent.ui.get_log_directory", return_value=log_dir):
            with patch("mini_agent.ui._open_directory_in_file_manager"):
                show_log_directory(open_file_manager=True)
        captured = capsys.readouterr()
        assert "Log Directory" in captured.out
        assert "test.log" in captured.out


class TestReadLogFile:
    def test_when_file_does_not_exist(self, capsys):
        with patch("mini_agent.ui.get_log_directory", return_value=Path("/nonexistent")):
            read_log_file("missing.log")
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_when_file_exists(self, capsys, tmp_path):
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        (log_dir / "test.log").write_text("log content here")
        with patch("mini_agent.ui.get_log_directory", return_value=log_dir):
            read_log_file("test.log")
        captured = capsys.readouterr()
        assert "log content here" in captured.out
        assert "End of file" in captured.out


class TestPrintImage:
    def test_returns_false_when_no_image_files(self):
        with patch("mini_agent.ui.Path") as mock_path_cls:
            mock_config_dir = MagicMock()
            mock_config_dir.__truediv__ = MagicMock(return_value=MagicMock())
            mock_path_cls.return_value.parent.__truediv__ = MagicMock(return_value=mock_config_dir)
            result = print_image()
        assert result is False


class TestOpenDirectoryInFileManager:
    @patch("mini_agent.ui.platform.system", return_value="Windows")
    @patch("mini_agent.ui.subprocess.run")
    def test_opens_on_windows(self, mock_run, mock_system):
        _open_directory_in_file_manager(Path("/some/dir"))
        mock_run.assert_called_once_with(["explorer", str(Path("/some/dir"))], check=False)

    @patch("mini_agent.ui.platform.system", return_value="Darwin")
    @patch("mini_agent.ui.subprocess.run")
    def test_opens_on_mac(self, mock_run, mock_system):
        _open_directory_in_file_manager(Path("/some/dir"))
        mock_run.assert_called_once_with(["open", str(Path("/some/dir"))], check=False)

    @patch("mini_agent.ui.platform.system", return_value="Linux")
    @patch("mini_agent.ui.subprocess.run")
    def test_opens_on_linux(self, mock_run, mock_system):
        _open_directory_in_file_manager(Path("/some/dir"))
        mock_run.assert_called_once_with(["xdg-open", str(Path("/some/dir"))], check=False)

    @patch("mini_agent.ui.platform.system", return_value="Linux")
    @patch("mini_agent.ui.subprocess.run", side_effect=FileNotFoundError)
    def test_handles_file_not_found(self, mock_run, mock_system, capsys):
        _open_directory_in_file_manager(Path("/some/dir"))
        captured = capsys.readouterr()
        assert "Could not open file manager" in captured.out

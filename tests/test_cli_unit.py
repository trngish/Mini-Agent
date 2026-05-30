from __future__ import annotations

from unittest.mock import MagicMock, patch

from mini_agent.cli import main, on_retry, parse_args


class TestOnRetry:
    @patch("mini_agent.retry.RetryConfig")
    def test_prints_retry_info(self, mock_retry_config_cls, capsys):
        mock_config = MagicMock()
        mock_config.calculate_delay.return_value = 2.0
        mock_retry_config_cls.return_value = mock_config

        exception = RuntimeError("timeout")
        on_retry(exception, 3)

        captured = capsys.readouterr()
        assert "尝试 3" in captured.out or "attempt 3" in captured.out
        assert "timeout" in captured.out
        assert "重试" in captured.out or "Retrying" in captured.out
        mock_config.calculate_delay.assert_called_once_with(2)

    @patch("mini_agent.retry.RetryConfig")
    def test_prints_attempt_number(self, mock_retry_config_cls, capsys):
        mock_config = MagicMock()
        mock_config.calculate_delay.return_value = 1.0
        mock_retry_config_cls.return_value = mock_config

        on_retry(ValueError("bad request"), 1)

        captured = capsys.readouterr()
        assert "尝试 1" in captured.out or "attempt 1" in captured.out
        assert "bad request" in captured.out
        mock_config.calculate_delay.assert_called_once_with(0)


class TestParseArgs:
    @patch("sys.argv", ["mini-agent"])
    def test_defaults(self):
        args = parse_args()
        assert args.workspace is None
        assert args.task is None
        assert args.api_key is None
        assert args.model is None
        assert args.provider is None
        assert args.no_skills is False
        assert args.no_mcp is False
        assert args.command is None

    @patch("sys.argv", ["mini-agent", "--workspace", "/tmp/project"])
    def test_workspace(self):
        args = parse_args()
        assert args.workspace == "/tmp/project"

    @patch("sys.argv", ["mini-agent", "-w", "/home/user/code"])
    def test_workspace_short(self):
        args = parse_args()
        assert args.workspace == "/home/user/code"

    @patch("sys.argv", ["mini-agent", "--task", "create a file"])
    def test_task(self):
        args = parse_args()
        assert args.task == "create a file"

    @patch("sys.argv", ["mini-agent", "-t", "run tests"])
    def test_task_short(self):
        args = parse_args()
        assert args.task == "run tests"

    @patch("sys.argv", ["mini-agent", "--api-key", "sk-abc123"])
    def test_api_key(self):
        args = parse_args()
        assert args.api_key == "sk-abc123"

    @patch("sys.argv", ["mini-agent", "--model", "claude-3-opus"])
    def test_model(self):
        args = parse_args()
        assert args.model == "claude-3-opus"

    @patch("sys.argv", ["mini-agent", "--provider", "anthropic"])
    def test_provider_anthropic(self):
        args = parse_args()
        assert args.provider == "anthropic"

    @patch("sys.argv", ["mini-agent", "--provider", "openai"])
    def test_provider_openai(self):
        args = parse_args()
        assert args.provider == "openai"

    @patch("sys.argv", ["mini-agent", "--no-skills"])
    def test_no_skills(self):
        args = parse_args()
        assert args.no_skills is True

    @patch("sys.argv", ["mini-agent", "--no-mcp"])
    def test_no_mcp(self):
        args = parse_args()
        assert args.no_mcp is True

    @patch("sys.argv", ["mini-agent", "log"])
    def test_log_command(self):
        args = parse_args()
        assert args.command == "log"
        assert args.filename is None

    @patch("sys.argv", ["mini-agent", "log", "session_2024.log"])
    def test_log_command_with_filename(self):
        args = parse_args()
        assert args.command == "log"
        assert args.filename == "session_2024.log"

    @patch(
        "sys.argv",
        [
            "mini-agent",
            "--workspace",
            "/tmp",
            "--task",
            "build",
            "--api-key",
            "sk-key",
            "--model",
            "gpt-4",
            "--provider",
            "openai",
            "--no-skills",
            "--no-mcp",
        ],
    )
    def test_combined_args(self):
        args = parse_args()
        assert args.workspace == "/tmp"
        assert args.task == "build"
        assert args.api_key == "sk-key"
        assert args.model == "gpt-4"
        assert args.provider == "openai"
        assert args.no_skills is True
        assert args.no_mcp is True


class TestMain:
    @patch("mini_agent.cli.parse_args")
    @patch("mini_agent.ui.read_log_file")
    def test_log_subcommand_with_filename(self, mock_read_log, mock_parse_args):
        mock_args = MagicMock()
        mock_args.command = "log"
        mock_args.filename = "test.log"
        mock_parse_args.return_value = mock_args

        main()

        mock_read_log.assert_called_once_with("test.log")

    @patch("mini_agent.cli.parse_args")
    @patch("mini_agent.ui.show_log_directory")
    def test_log_subcommand_without_filename(self, mock_show_log, mock_parse_args):
        mock_args = MagicMock()
        mock_args.command = "log"
        mock_args.filename = None
        mock_parse_args.return_value = mock_args

        main()

        mock_show_log.assert_called_once_with(open_file_manager=True)

    @patch("mini_agent.cli.parse_args")
    @patch("mini_agent.cli.asyncio")
    @patch("mini_agent.cli.CLIOverrideConfig")
    @patch("mini_agent.cli.Path")
    def test_run_agent_called_with_defaults(self, mock_path, mock_cli_override, mock_asyncio, mock_parse_args):
        mock_args = MagicMock()
        mock_args.command = None
        mock_args.workspace = None
        mock_args.task = None
        mock_args.api_key = None
        mock_args.api_base = None
        mock_args.model = None
        mock_args.provider = None
        mock_args.max_steps = None
        mock_args.platform = None
        mock_args.no_skills = False
        mock_args.no_mcp = False
        mock_parse_args.return_value = mock_args

        mock_path.cwd.return_value.resolve.return_value = "/cwd"

        main()

        mock_asyncio.run.assert_called_once()

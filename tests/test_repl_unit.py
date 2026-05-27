from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mini_agent.repl import InteractiveLoop
from mini_agent.schema import AgentMode


def _make_mock_agent(mode=AgentMode.AGENT, messages=None, api_call_count=0, workspace_dir="."):
    agent = MagicMock()
    agent.mode = mode
    agent.messages = messages if messages is not None else [MagicMock()]
    agent.api_call_count = api_call_count
    agent.workspace_dir = Path(workspace_dir)
    agent.tools = {}
    agent.tool_list = []
    agent.add_user_message = MagicMock()
    agent.run = AsyncMock()
    agent._session_manager = MagicMock()
    agent.save_session = MagicMock(return_value="session_001")
    agent.load_session = MagicMock()
    agent.list_sessions = MagicMock(return_value=[])
    agent.llm = MagicMock()
    agent.logger = MagicMock()
    agent.logger.get_log_file_path = MagicMock(return_value="/tmp/test.log")
    return agent


def _make_loop(agent=None, workspace_dir=None, config=None, skill_loader=None, m27_config=None):
    if agent is None:
        agent = _make_mock_agent()
    if workspace_dir is None:
        workspace_dir = Path("/tmp/workspace")
    if config is None:
        config = MagicMock()
    return InteractiveLoop(
        agent=agent,
        workspace_dir=workspace_dir,
        config=config,
        skill_loader=skill_loader,
        m27_config=m27_config,
    )


def _find_binding(kb, key_suffix):
    for b in kb.bindings:
        if any(key_suffix in str(k) for k in b.keys):
            return b
    return None


class TestInteractiveLoopInit:
    def test_stores_agent(self):
        agent = _make_mock_agent()
        loop = _make_loop(agent=agent)
        assert loop.agent is agent

    def test_stores_workspace_dir(self):
        ws = Path("/tmp/project")
        loop = _make_loop(workspace_dir=ws)
        assert loop.workspace_dir == ws

    def test_stores_config(self):
        config = MagicMock()
        config.llm = MagicMock()
        config.llm.model = "test-model"
        loop = _make_loop(config=config)
        assert loop.config is config

    def test_stores_skill_loader(self):
        loader = MagicMock()
        loop = _make_loop(skill_loader=loader)
        assert loop.skill_loader is loader

    def test_skill_loader_defaults_to_none(self):
        loop = _make_loop()
        assert loop.skill_loader is None

    def test_stores_m27_config(self):
        m27 = {"token_limit": 80000}
        loop = _make_loop(m27_config=m27)
        assert loop.m27_config == m27

    def test_m27_config_defaults_to_empty_dict(self):
        loop = _make_loop()
        assert loop.m27_config == {}

    def test_session_start_is_datetime(self):
        loop = _make_loop()
        assert isinstance(loop.session_start, datetime)

    def test_session_start_is_recent(self):
        before = datetime.now()
        loop = _make_loop()
        after = datetime.now()
        assert before <= loop.session_start <= after


class TestBuildKeyBindings:
    def test_returns_key_bindings(self):
        from prompt_toolkit.key_binding import KeyBindings

        loop = _make_loop()
        kb = loop._build_key_bindings()
        assert isinstance(kb, KeyBindings)

    def test_has_four_bindings(self):
        loop = _make_loop()
        kb = loop._build_key_bindings()
        assert len(kb.bindings) == 4

    def test_binding_keys_present(self):
        loop = _make_loop()
        kb = loop._build_key_bindings()
        key_names = [tuple(str(k) for k in b.keys) for b in kb.bindings]
        assert any("ControlU" in k[0] for k in key_names)
        assert any("ControlL" in k[0] for k in key_names)
        assert any("ControlJ" in k[0] for k in key_names)
        assert any("ControlI" in k[0] for k in key_names)

    def test_c_u_resets_buffer(self):
        loop = _make_loop()
        kb = loop._build_key_bindings()
        binding = _find_binding(kb, "ControlU")
        assert binding is not None
        mock_event = MagicMock()
        binding.handler(mock_event)
        mock_event.current_buffer.reset.assert_called_once()

    def test_c_l_clears_screen(self):
        loop = _make_loop()
        kb = loop._build_key_bindings()
        binding = _find_binding(kb, "ControlL")
        assert binding is not None
        mock_event = MagicMock()
        binding.handler(mock_event)
        mock_event.app.renderer.clear.assert_called_once()

    def test_c_j_inserts_newline(self):
        loop = _make_loop()
        kb = loop._build_key_bindings()
        binding = _find_binding(kb, "ControlJ")
        assert binding is not None
        mock_event = MagicMock()
        binding.handler(mock_event)
        mock_event.current_buffer.insert_text.assert_called_once_with("\n")

    def test_tab_cycles_mode_from_plan(self):
        agent = _make_mock_agent(mode=AgentMode.PLAN)
        loop = _make_loop(agent=agent)
        kb = loop._build_key_bindings()
        binding = _find_binding(kb, "ControlI")
        assert binding is not None
        mock_event = MagicMock()
        binding.handler(mock_event)
        assert agent.mode == AgentMode.AGENT

    def test_tab_cycles_mode_from_agent(self):
        agent = _make_mock_agent(mode=AgentMode.AGENT)
        loop = _make_loop(agent=agent)
        kb = loop._build_key_bindings()
        binding = _find_binding(kb, "ControlI")
        mock_event = MagicMock()
        binding.handler(mock_event)
        assert agent.mode == AgentMode.YOLO

    def test_tab_cycles_mode_from_yolo_back_to_plan(self):
        agent = _make_mock_agent(mode=AgentMode.YOLO)
        loop = _make_loop(agent=agent)
        kb = loop._build_key_bindings()
        binding = _find_binding(kb, "ControlI")
        mock_event = MagicMock()
        binding.handler(mock_event)
        assert agent.mode == AgentMode.PLAN


class TestBuildCompleter:
    def test_completer_has_expected_commands(self):
        from prompt_toolkit.completion import WordCompleter

        expected = [
            "/help",
            "/clear",
            "/history",
            "/stats",
            "/log",
            "/mode",
            "/save",
            "/load",
            "/list",
            "/subagent",
            "/skills",
            "/brainstorm",
            "/plan",
            "/debug",
            "/exit",
        ]
        completer = WordCompleter(expected, ignore_case=True)
        for cmd in expected:
            assert cmd in completer.words

    def test_completer_ignores_case(self):
        from prompt_toolkit.completion import WordCompleter

        completer = WordCompleter(["/help", "/exit"], ignore_case=True)
        assert completer.ignore_case is True

    def test_completer_is_word_completer(self):
        from prompt_toolkit.completion import WordCompleter

        completer = WordCompleter(["/help"], ignore_case=True)
        assert isinstance(completer, WordCompleter)

    def test_completer_words_match_repl_commands(self):
        from prompt_toolkit.completion import WordCompleter

        repl_commands = [
            "/help",
            "/clear",
            "/history",
            "/stats",
            "/log",
            "/mode",
            "/save",
            "/load",
            "/list",
            "/subagent",
            "/skills",
            "/brainstorm",
            "/plan",
            "/debug",
            "/exit",
        ]
        completer = WordCompleter(repl_commands, ignore_case=True)
        assert len(completer.words) == 15
        assert "/help" in completer.words
        assert "/exit" in completer.words
        assert "/debug" in completer.words


class TestDispatchCommand:
    async def test_help_calls_print_help(self):
        loop = _make_loop()
        with patch("mini_agent.repl.print_help") as mock_help:
            result = await loop._dispatch_command("/help", "/help")
        mock_help.assert_called_once()
        assert result is True

    async def test_clear_resets_messages(self):
        agent = _make_mock_agent(messages=[MagicMock(), MagicMock(), MagicMock()])
        loop = _make_loop(agent=agent)
        result = await loop._dispatch_command("/clear", "/clear")
        assert len(agent.messages) == 1
        assert result is True

    async def test_history_prints_message_count(self, capsys):
        agent = _make_mock_agent(messages=[MagicMock(), MagicMock(), MagicMock()])
        loop = _make_loop(agent=agent)
        result = await loop._dispatch_command("/history", "/history")
        captured = capsys.readouterr()
        assert "3" in captured.out
        assert result is True

    async def test_stats_calls_print_stats(self):
        loop = _make_loop()
        with patch("mini_agent.repl.print_stats") as mock_stats:
            result = await loop._dispatch_command("/stats", "/stats")
        mock_stats.assert_called_once()
        assert result is True

    async def test_mode_sets_agent_mode(self, capsys):
        agent = _make_mock_agent(mode=AgentMode.PLAN)
        loop = _make_loop(agent=agent)
        result = await loop._dispatch_command("/mode agent", "/mode agent")
        assert agent.mode == AgentMode.AGENT
        captured = capsys.readouterr()
        assert "AGENT" in captured.out
        assert result is True

    async def test_mode_yolo(self, capsys):
        agent = _make_mock_agent(mode=AgentMode.AGENT)
        loop = _make_loop(agent=agent)
        result = await loop._dispatch_command("/mode yolo", "/mode yolo")
        assert agent.mode == AgentMode.YOLO
        assert result is True

    async def test_mode_plan(self, capsys):
        agent = _make_mock_agent(mode=AgentMode.YOLO)
        loop = _make_loop(agent=agent)
        result = await loop._dispatch_command("/mode plan", "/mode plan")
        assert agent.mode == AgentMode.PLAN
        assert result is True

    async def test_mode_without_arg_does_nothing(self, capsys):
        agent = _make_mock_agent(mode=AgentMode.AGENT)
        loop = _make_loop(agent=agent)
        result = await loop._dispatch_command("/mode", "/mode")
        assert agent.mode == AgentMode.AGENT
        assert result is True

    async def test_exit_returns_false(self, capsys):
        loop = _make_loop()
        with patch("mini_agent.repl.print_stats"):
            result = await loop._dispatch_command("/exit", "/exit")
        assert result is False

    async def test_quit_returns_false(self, capsys):
        loop = _make_loop()
        with patch("mini_agent.repl.print_stats"):
            result = await loop._dispatch_command("/quit", "/quit")
        assert result is False

    async def test_q_returns_false(self, capsys):
        loop = _make_loop()
        with patch("mini_agent.repl.print_stats"):
            result = await loop._dispatch_command("/q", "/q")
        assert result is False

    async def test_exit_prints_goodbye(self, capsys):
        loop = _make_loop()
        with patch("mini_agent.repl.print_stats"):
            await loop._dispatch_command("/exit", "/exit")
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    async def test_log_without_arg_shows_directory(self):
        loop = _make_loop()
        with patch("mini_agent.repl.show_log_directory") as mock_show:
            result = await loop._dispatch_command("/log", "/log")
        mock_show.assert_called_once_with(open_file_manager=True)
        assert result is True

    async def test_log_with_arg_reads_file(self):
        loop = _make_loop()
        with patch("mini_agent.repl.read_log_file") as mock_read:
            result = await loop._dispatch_command("/log test.log", "/log test.log")
        mock_read.assert_called_once_with("test.log")
        assert result is True

    async def test_log_with_quoted_arg(self):
        loop = _make_loop()
        with patch("mini_agent.repl.read_log_file") as mock_read:
            result = await loop._dispatch_command('/log "my log.log"', '/log "my log.log"')
        mock_read.assert_called_once_with("my log.log")
        assert result is True

    async def test_save_calls_agent_save_session(self, capsys):
        agent = _make_mock_agent()
        loop = _make_loop(agent=agent)
        result = await loop._dispatch_command("/save my session", "/save my session")
        agent.save_session.assert_called_once_with("my session")
        assert result is True

    async def test_save_prints_session_id(self, capsys):
        agent = _make_mock_agent()
        agent.save_session.return_value = "session_abc"
        loop = _make_loop(agent=agent)
        result = await loop._dispatch_command("/save", "/save")
        captured = capsys.readouterr()
        assert "session_abc" in captured.out
        assert result is True

    async def test_load_calls_agent_load_session(self):
        agent = _make_mock_agent()
        loop = _make_loop(agent=agent)
        result = await loop._dispatch_command("/load session_001", "/load session_001")
        agent.load_session.assert_called_once_with("session_001")
        assert result is True

    async def test_load_without_arg_does_nothing(self):
        agent = _make_mock_agent()
        loop = _make_loop(agent=agent)
        result = await loop._dispatch_command("/load", "/load")
        agent.load_session.assert_not_called()
        assert result is True

    async def test_list_calls_agent_list_sessions(self, capsys):
        agent = _make_mock_agent()
        agent.list_sessions.return_value = [
            {"id": "s1", "created": "2025-01-01T10:00:00", "label": "test"},
        ]
        loop = _make_loop(agent=agent)
        result = await loop._dispatch_command("/list", "/list")
        agent.list_sessions.assert_called_once()
        captured = capsys.readouterr()
        assert "s1" in captured.out
        assert result is True

    async def test_list_limits_to_ten(self, capsys):
        agent = _make_mock_agent()
        sessions = [{"id": f"s{i}", "created": "2025-01-01T10:00:00", "label": ""} for i in range(15)]
        agent.list_sessions.return_value = sessions
        loop = _make_loop(agent=agent)
        result = await loop._dispatch_command("/list", "/list")
        captured = capsys.readouterr()
        assert "s0" in captured.out
        assert "s9" in captured.out
        assert result is True

    async def test_skills_with_loader(self, capsys):
        loader = MagicMock()
        skill_mock = MagicMock()
        skill_mock.description = "A brainstorming skill for creative work"
        loader.list_skills.return_value = ["brainstorming"]
        loader.get_skill.return_value = skill_mock
        loop = _make_loop(skill_loader=loader)
        result = await loop._dispatch_command("/skills", "/skills")
        loader.list_skills.assert_called_once()
        captured = capsys.readouterr()
        assert "brainstorming" in captured.out
        assert result is True

    async def test_skills_without_loader(self, capsys):
        loop = _make_loop(skill_loader=None)
        result = await loop._dispatch_command("/skills", "/skills")
        captured = capsys.readouterr()
        assert "Skills not loaded" in captured.out
        assert result is True

    async def test_subagent_with_task(self):
        agent = _make_mock_agent()
        loop = _make_loop(agent=agent, m27_config={"key": "val"})
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.elapsed = 1.5
        mock_result.content = "done"
        with patch("mini_agent.repl.SubAgent") as mock_subagent_cls:
            mock_subagent_instance = MagicMock()
            mock_subagent_instance.run = AsyncMock(return_value=mock_result)
            mock_subagent_cls.return_value = mock_subagent_instance
            result = await loop._dispatch_command("/subagent do something", "/subagent do something")
        mock_subagent_cls.assert_called_once_with(
            llm_client=agent.llm,
            tools=list(agent.tools.values()),
            m27_config={"key": "val"},
        )
        mock_subagent_instance.run.assert_called_once_with("do something")
        assert result is True

    async def test_subagent_without_task_does_nothing(self):
        agent = _make_mock_agent()
        loop = _make_loop(agent=agent)
        with patch("mini_agent.repl.SubAgent") as mock_subagent_cls:
            result = await loop._dispatch_command("/subagent", "/subagent")
        mock_subagent_cls.assert_not_called()
        assert result is True

    async def test_subagent_failure(self, capsys):
        agent = _make_mock_agent()
        loop = _make_loop(agent=agent)
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "something went wrong"
        with patch("mini_agent.repl.SubAgent") as mock_subagent_cls:
            mock_subagent_instance = MagicMock()
            mock_subagent_instance.run = AsyncMock(return_value=mock_result)
            mock_subagent_cls.return_value = mock_subagent_instance
            result = await loop._dispatch_command("/subagent fail task", "/subagent fail task")
        captured = capsys.readouterr()
        assert "something went wrong" in captured.out
        assert result is True

    async def test_subagent_exception(self, capsys):
        agent = _make_mock_agent()
        loop = _make_loop(agent=agent)
        with patch("mini_agent.repl.SubAgent") as mock_subagent_cls:
            mock_subagent_instance = MagicMock()
            mock_subagent_instance.run = AsyncMock(side_effect=RuntimeError("boom"))
            mock_subagent_cls.return_value = mock_subagent_instance
            result = await loop._dispatch_command("/subagent crash", "/subagent crash")
        captured = capsys.readouterr()
        assert "boom" in captured.out
        assert result is True

    async def test_brainstorm_with_skill(self, capsys):
        loader = MagicMock()
        loader.get_skill.return_value = MagicMock()
        loop = _make_loop(skill_loader=loader)
        result = await loop._dispatch_command("/brainstorm", "/brainstorm")
        captured = capsys.readouterr()
        assert "Brainstorming" in captured.out
        assert result is True

    async def test_brainstorm_without_skill(self, capsys):
        loader = MagicMock()
        loader.get_skill.return_value = None
        loop = _make_loop(skill_loader=loader)
        result = await loop._dispatch_command("/brainstorm", "/brainstorm")
        captured = capsys.readouterr()
        assert "not found" in captured.out
        assert result is True

    async def test_brainstorm_without_loader(self, capsys):
        loop = _make_loop(skill_loader=None)
        result = await loop._dispatch_command("/brainstorm", "/brainstorm")
        captured = capsys.readouterr()
        assert "not found" in captured.out
        assert result is True

    async def test_plan_with_skill(self, capsys):
        loader = MagicMock()
        loader.get_skill.return_value = MagicMock()
        loop = _make_loop(skill_loader=loader)
        result = await loop._dispatch_command("/plan", "/plan")
        captured = capsys.readouterr()
        assert "Writing Plans" in captured.out
        assert result is True

    async def test_plan_without_skill(self, capsys):
        loader = MagicMock()
        loader.get_skill.return_value = None
        loop = _make_loop(skill_loader=loader)
        result = await loop._dispatch_command("/plan", "/plan")
        captured = capsys.readouterr()
        assert "not found" in captured.out
        assert result is True

    async def test_plan_without_loader(self, capsys):
        loop = _make_loop(skill_loader=None)
        result = await loop._dispatch_command("/plan", "/plan")
        captured = capsys.readouterr()
        assert "not found" in captured.out
        assert result is True

    async def test_unknown_command(self, capsys):
        loop = _make_loop()
        result = await loop._dispatch_command("/unknown", "/unknown")
        captured = capsys.readouterr()
        assert "Unknown" in captured.out
        assert result is True

    async def test_debug_prints_log_path(self, capsys):
        loop = _make_loop()
        result = await loop._dispatch_command("/debug", "/debug")
        captured = capsys.readouterr()
        assert "/tmp/test.log" in captured.out
        assert result is True


class TestHandleModeSwitch:
    def test_plan_to_agent(self):
        agent = _make_mock_agent(mode=AgentMode.PLAN)
        loop = _make_loop(agent=agent)
        kb = loop._build_key_bindings()
        binding = _find_binding(kb, "ControlI")
        mock_event = MagicMock()
        binding.handler(mock_event)
        assert agent.mode == AgentMode.AGENT

    def test_agent_to_yolo(self):
        agent = _make_mock_agent(mode=AgentMode.AGENT)
        loop = _make_loop(agent=agent)
        kb = loop._build_key_bindings()
        binding = _find_binding(kb, "ControlI")
        mock_event = MagicMock()
        binding.handler(mock_event)
        assert agent.mode == AgentMode.YOLO

    def test_yolo_to_plan(self):
        agent = _make_mock_agent(mode=AgentMode.YOLO)
        loop = _make_loop(agent=agent)
        kb = loop._build_key_bindings()
        binding = _find_binding(kb, "ControlI")
        mock_event = MagicMock()
        binding.handler(mock_event)
        assert agent.mode == AgentMode.PLAN

    def test_full_cycle_plan_agent_yolo_plan(self):
        agent = _make_mock_agent(mode=AgentMode.PLAN)
        loop = _make_loop(agent=agent)
        kb = loop._build_key_bindings()
        binding = _find_binding(kb, "ControlI")
        mock_event = MagicMock()
        binding.handler(mock_event)
        assert agent.mode == AgentMode.AGENT
        binding.handler(mock_event)
        assert agent.mode == AgentMode.YOLO
        binding.handler(mock_event)
        assert agent.mode == AgentMode.PLAN

    def test_cycle_prints_mode_name(self, capsys):
        agent = _make_mock_agent(mode=AgentMode.PLAN)
        loop = _make_loop(agent=agent)
        kb = loop._build_key_bindings()
        binding = _find_binding(kb, "ControlI")
        mock_event = MagicMock()
        binding.handler(mock_event)
        captured = capsys.readouterr()
        assert "AGENT" in captured.out

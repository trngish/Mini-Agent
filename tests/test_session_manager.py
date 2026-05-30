from mini_agent.schema import Message
from mini_agent.session import SessionManager


class TestSessionManager:
    def test_init(self, tmp_path):
        sm = SessionManager(session_dir=tmp_path)
        assert sm.session_dir == tmp_path

    def test_default_session_dir(self):
        sm = SessionManager()
        assert sm.session_dir.exists()

    def test_save_and_load_session(self, tmp_path):
        sm = SessionManager(session_dir=tmp_path)
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        session_id = sm.save(messages)
        assert session_id is not None
        loaded_messages, loaded_result, loaded_state = sm.load(session_id)
        assert loaded_messages is not None
        assert len(loaded_messages) == 3
        assert loaded_messages[0].role == "system"
        assert loaded_messages[1].content == "Hello"
        assert loaded_result is None  # No result saved yet

    def test_save_and_load_with_result(self, tmp_path):
        """Test saving and loading session with result."""
        sm = SessionManager(session_dir=tmp_path)
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        result = "Task completed successfully"
        session_id = sm.save(messages, result=result)
        loaded_messages, loaded_result, loaded_state = sm.load(session_id)
        assert loaded_messages is not None
        assert len(loaded_messages) == 2
        assert loaded_result == result

    def test_load_nonexistent_session(self, tmp_path):
        sm = SessionManager(session_dir=tmp_path)
        messages, result, state = sm.load("nonexistent-id")
        assert messages is None
        assert result is None

    def test_list_sessions_empty(self, tmp_path):
        sm = SessionManager(session_dir=tmp_path)
        sessions = sm.list_sessions()
        assert isinstance(sessions, list)
        assert len(sessions) == 0

    def test_list_sessions_after_save(self, tmp_path):
        sm = SessionManager(session_dir=tmp_path)
        messages = [Message(role="user", content="test")]
        sm.save(messages)
        sessions = sm.list_sessions()
        assert len(sessions) >= 1

    def test_save_with_label(self, tmp_path):
        sm = SessionManager(session_dir=tmp_path)
        messages = [Message(role="user", content="test")]
        session_id = sm.save(messages, label="my session")
        loaded = sm.load(session_id)
        assert loaded is not None

    def test_delete_session(self, tmp_path):
        sm = SessionManager(session_dir=tmp_path)
        messages = [Message(role="user", content="test")]
        session_id = sm.save(messages)
        result = sm.delete(session_id)
        assert result is True
        loaded_messages, loaded_result, loaded_state = sm.load(session_id)
        assert loaded_messages is None
        assert loaded_result is None

    def test_delete_nonexistent_session(self, tmp_path):
        sm = SessionManager(session_dir=tmp_path)
        result = sm.delete("nonexistent-id")
        assert result is False

    def test_clear_index(self, tmp_path):
        sm = SessionManager(session_dir=tmp_path)
        messages = [Message(role="user", content="test")]
        sm.save(messages)
        sm.clear_index()
        assert sm._index is None
        assert sm._index_loaded is False

    def test_multiple_sessions(self, tmp_path):
        sm = SessionManager(session_dir=tmp_path)
        sm.save([Message(role="user", content="first")])
        sm.save([Message(role="user", content="second")])
        sessions = sm.list_sessions()
        assert len(sessions) == 2

    def test_index_trim(self, tmp_path):
        sm = SessionManager(session_dir=tmp_path)
        sm.MAX_SESSIONS_IN_INDEX = 3
        for i in range(5):
            sm.save([Message(role="user", content=f"msg {i}")])
        sm._ensure_index_loaded()
        assert len(sm._index) <= 3

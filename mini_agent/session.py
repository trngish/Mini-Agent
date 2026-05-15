"""Session save/resume manager."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .schema import FunctionCall, Message, ToolCall


class SessionManager:
    """Manages session persistence (save/resume/list)."""

    def __init__(self, session_dir: Optional[Path] = None):
        self.session_dir = session_dir or Path.home() / ".mini-agent" / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def _serialize_messages(self, messages: list[Message]) -> list[dict]:
        return [msg.model_dump() for msg in messages]

    def _deserialize_messages(self, data: list[dict]) -> list[Message]:
        return [Message(**msg) for msg in data]

    def save(self, messages: list[Message], label: str = "") -> str:
        """Save messages to a session file. Returns session ID."""
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        data = {
            "id": session_id,
            "label": label,
            "created": timestamp,
            "messages": self._serialize_messages(messages),
        }
        path = self.session_dir / f"{session_id}.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return session_id

    def load(self, session_id: str) -> Optional[list[Message]]:
        """Load messages from a session file."""
        path = self.session_dir / f"{session_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return self._deserialize_messages(data["messages"])

    def list_sessions(self) -> list[dict]:
        """List all saved sessions."""
        sessions = []
        for path in sorted(self.session_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append({
                    "id": data.get("id", path.stem),
                    "label": data.get("label", ""),
                    "created": data.get("created", ""),
                    "messages": len(data.get("messages", [])),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a session file."""
        path = self.session_dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

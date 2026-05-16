"""Agent run logger with rotation and compression support.

Responsible for recording the complete interaction process of each agent run, including:
- LLM requests and responses
- Tool calls and results
- Log rotation based on size or date
- Compression of old logs with gzip
"""

import asyncio
import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .schema import Message, ToolCall


class AgentLogger:
    """Agent run logger with log rotation and compression support.

    Logs are stored in ~/.mini-agent/log/ directory with:
    - Automatic rotation when file exceeds MAX_SIZE bytes
    - Automatic cleanup of logs older than MAX_AGE days
    - Gzip compression for rotated logs
    - Optional async write for better performance
    - JSON format for easy parsing
    """

    # Log rotation settings
    MAX_LOG_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB per file
    MAX_LOG_AGE_DAYS: int = 7  # Keep logs for 7 days
    LOG_DIR: Path = Path.home() / ".mini-agent" / "log"
    # Async write settings
    ASYNC_WRITE_ENABLED: bool = True
    _write_queue: asyncio.Queue | None = None
    _writer_task: asyncio.Task | None = None

    def __init__(self):
        """Initialize logger with rotation settings."""
        self.log_dir = self.LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file: Path | None = None
        self.log_index = 0
        self._current_size = 0
        self._rotation_check_done = False
        # Initialize async write queue for better performance
        self._write_queue: asyncio.Queue[str] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._shutdown_event = False

    def _should_rotate(self) -> bool:
        """Check if log rotation is needed."""
        if self.log_file is None:
            return False
        return self._current_size >= self.MAX_LOG_SIZE_BYTES

    def _rotate_log(self) -> None:
        """Rotate log file if size limit exceeded, with optional compression."""
        if self.log_file is None:
            return
        
        # Check file size
        if self.log_file.exists():
            self._current_size = self.log_file.stat().st_size
        else:
            self._current_size = 0
        
        if self._current_size < self.MAX_LOG_SIZE_BYTES:
            return
        
        # Rename current log with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"{self.log_file.stem}_{timestamp}.rotated"
        rotated_path = self.log_dir / rotated_name
        
        try:
            self.log_file.rename(rotated_path)
            # Compress the rotated log in background
            self._compress_log_async(rotated_path)
        except OSError:
            # If rename fails, just delete and start fresh
            self.log_file.unlink(missing_ok=True)
        
        self.log_file = None
        self.log_index = 0
        self._current_size = 0

    def _compress_log_async(self, log_path: Path) -> None:
        """Compress a log file using gzip in a background thread.
        
        Args:
            log_path: Path to the log file to compress
        """
        try:
            compressed_path = log_path.with_suffix(".log.gz")
            with open(log_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    f_out.write(f_in.read())
            # Remove original after successful compression
            log_path.unlink(missing_ok=True)
        except Exception:
            # Compression failed, keep original
            pass

    def _cleanup_old_logs(self) -> None:
        """Remove log files older than MAX_LOG_AGE_DAYS."""
        if not self.log_dir.exists():
            return
        
        cutoff = datetime.now() - timedelta(days=self.MAX_LOG_AGE_DAYS)
        
        for path in self.log_dir.glob("*.log"):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
        
        # Also clean up rotated files
        for path in self.log_dir.glob("*.rotated"):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue

    def _ensure_log_file(self) -> None:
        """Ensure log file exists and is ready for writing."""
        if self._rotation_check_done and self.log_file is not None:
            return
        
        # Run cleanup on first log access
        if not self._rotation_check_done:
            self._cleanup_old_logs()
            self._rotation_check_done = True
        
        if self.log_file is None:
            self.start_new_run()

    def start_new_run(self) -> None:
        """Start new run, create new log file."""
        # Check rotation before creating new file
        if self.log_file is not None:
            self._rotate_log()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_filename = f"agent_run_{timestamp}.log"
        self.log_file = self.log_dir / log_filename
        self.log_index = 0
        self._current_size = 0

        # Write log header
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Agent Run Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        self._current_size = self.log_file.stat().st_size

    def log_request(self, messages: list[Message], tools: list[Any] | None = None) -> None:
        """Log LLM request.

        Args:
            messages: Message list
            tools: Tool list (optional)
        """
        self._ensure_log_file()
        self.log_index += 1

        # Build complete request data structure
        request_data = {
            "messages": [],
            "tools": [],
        }

        # Convert messages to JSON serializable format
        for msg in messages:
            msg_dict: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.thinking:
                msg_dict["thinking"] = msg.thinking
            if msg.tool_calls:
                msg_dict["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.name:
                msg_dict["name"] = msg.name

            request_data["messages"].append(msg_dict)

        # Only record tool names
        if tools:
            request_data["tools"] = [tool.name for tool in tools]

        # Format as JSON
        content = "LLM Request:\n\n"
        content += json.dumps(request_data, indent=2, ensure_ascii=False)

        self._write_log("REQUEST", content)

    def log_response(
        self,
        content: str,
        thinking: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        finish_reason: str | None = None,
    ) -> None:
        """Log LLM response.

        Args:
            content: Response content
            thinking: Thinking content (optional)
            tool_calls: Tool call list (optional)
            finish_reason: Finish reason (optional)
        """
        self._ensure_log_file()
        self.log_index += 1

        # Build complete response data structure
        response_data: dict[str, Any] = {
            "content": content,
        }

        if thinking:
            response_data["thinking"] = thinking

        if tool_calls:
            response_data["tool_calls"] = [tc.model_dump() for tc in tool_calls]

        if finish_reason:
            response_data["finish_reason"] = finish_reason

        # Format as JSON
        log_content = "LLM Response:\n\n"
        log_content += json.dumps(response_data, indent=2, ensure_ascii=False)

        self._write_log("RESPONSE", log_content)

    def log_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result_success: bool,
        result_content: str | None = None,
        result_error: str | None = None,
    ) -> None:
        """Log tool execution result.

        Args:
            tool_name: Tool name
            arguments: Tool arguments
            result_success: Whether successful
            result_content: Result content (on success)
            result_error: Error message (on failure)
        """
        self._ensure_log_file()
        self.log_index += 1

        # Build complete tool execution result data structure
        tool_result_data: dict[str, Any] = {
            "tool_name": tool_name,
            "arguments": arguments,
            "success": result_success,
        }

        if result_success:
            tool_result_data["result"] = result_content
        else:
            tool_result_data["error"] = result_error

        # Format as JSON
        content = "Tool Execution:\n\n"
        content += json.dumps(tool_result_data, indent=2, ensure_ascii=False)

        self._write_log("TOOL_RESULT", content)

    def _write_log(self, log_type: str, content: str) -> None:
        """Write log entry.

        Args:
            log_type: Log type (REQUEST, RESPONSE, TOOL_RESULT)
            content: Log content
        """
        if self.log_file is None:
            return

        entry = "\n" + "-" * 80 + "\n"
        entry += f"[{self.log_index}] {log_type}\n"
        entry += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
        entry += "-" * 80 + "\n"
        entry += content + "\n"

        if self.ASYNC_WRITE_ENABLED and self._write_queue is not None:
            # Queue for async write
            self._write_queue.put_nowait(entry)
        else:
            # Synchronous write
            self._write_log_sync(entry)

    def _write_log_sync(self, entry: str) -> None:
        """Synchronously write log entry to file.
        
        Args:
            entry: Log entry string
        """
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(entry)
        
        self._current_size += len(entry.encode("utf-8"))

    def get_log_file_path(self) -> Path:
        """Get current log file path."""
        return self.log_file or self.LOG_DIR / "placeholder.log"
    
    def flush(self) -> None:
        """Flush any pending writes to disk."""
        if self._write_queue is not None:
            # Drain the queue synchronously
            while not self._write_queue.empty():
                try:
                    entry = self._write_queue.get_nowait()
                    self._write_log_sync(entry)
                except asyncio.QueueEmpty:
                    break
        if self.log_file is not None:
            # Open and close to flush
            with open(self.log_file, "a", encoding="utf-8"):
                pass
    
    def get_log_stats(self) -> dict[str, Any]:
        """Get statistics about logs.
        
        Returns:
            Dict with log statistics
        """
        if not self.log_dir.exists():
            return {"total_logs": 0, "total_size": 0, "oldest_log": None, "newest_log": None}
        
        log_files = list(self.log_dir.glob("*.log"))
        rotated_files = list(self.log_dir.glob("*.rotated"))
        compressed_files = list(self.log_dir.glob("*.log.gz"))
        
        total_size = sum(f.stat().st_size for f in log_files)
        compressed_size = sum(f.stat().st_size for f in compressed_files)
        
        oldest = None
        newest = None
        for f in log_files:
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                if oldest is None or mtime < oldest:
                    oldest = mtime
                if newest is None or mtime > newest:
                    newest = mtime
            except OSError:
                continue
        
        return {
            "total_logs": len(log_files),
            "rotated_logs": len(rotated_files),
            "compressed_logs": len(compressed_files),
            "total_size": total_size,
            "compressed_size": compressed_size,
            "oldest_log": oldest.isoformat() if oldest else None,
            "newest_log": newest.isoformat() if newest else None,
            "max_age_days": self.MAX_LOG_AGE_DAYS,
            "max_size_mb": self.MAX_LOG_SIZE_BYTES / (1024 * 1024),
        }
"""支持日志轮转和压缩的智能体运行日志记录器。

负责记录每次智能体运行的完整交互过程，包括：
- LLM 请求和响应
- 工具调用及结果
- 基于大小或日期的日志轮转
- 使用 gzip 压缩旧日志
"""

import asyncio
import gzip
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .schema import Message, ToolCall


class AgentLogger:
    """支持日志轮转和压缩的智能体运行日志记录器。

    日志存储在 ~/.mini-agent/log/ 目录，具有以下特性：
    - 当文件超过 MAX_SIZE 字节时自动轮转
    - 自动清理超过 MAX_AGE 天的日志
    - 对轮转的日志进行 gzip 压缩
    - 可选的异步写入以提升性能
    - JSON 格式便于解析
    """

    # 日志轮转设置
    MAX_LOG_SIZE_BYTES: int = 10 * 1024 * 1024  # 每个文件 10MB
    MAX_LOG_AGE_DAYS: int = 7  # 保留 7 天的日志
    LOG_DIR: Path = Path.home() / ".mini-agent" / "log"
    # 异步写入设置
    ASYNC_WRITE_ENABLED: bool = True
    MAX_WRITE_QUEUE_SIZE: int = 1000  # 最大队列大小，防止内存问题

    def __init__(self) -> None:
        """初始化日志记录器，设置轮转参数。"""
        self.log_dir = self.LOG_DIR
        self._log_disabled = False
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            self._log_disabled = True
        self.log_file: Path | None = None
        self.log_index = 0
        self._current_size = 0
        self._rotation_check_done = False
        # 初始化异步写入队列，设置最大大小以提升性能
        self._write_queue: asyncio.Queue[str] | None = None
        self._writer_task: asyncio.Task[None] | None = None
        self._shutdown_event = False
        self._max_queue_size = self.MAX_WRITE_QUEUE_SIZE

    def _should_rotate(self) -> bool:
        """检查是否需要日志轮转。"""
        if self.log_file is None:
            return False
        return self._current_size >= self.MAX_LOG_SIZE_BYTES

    def _rotate_log(self) -> None:
        """如果文件大小超出限制则轮转日志，可选择压缩。"""
        if self.log_file is None:
            return

        # 检查文件大小
        if self.log_file.exists():
            self._current_size = self.log_file.stat().st_size
        else:
            self._current_size = 0

        if self._current_size < self.MAX_LOG_SIZE_BYTES:
            return

        # 使用时间戳重命名当前日志
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_name = f"{self.log_file.stem}_{timestamp}.rotated"
        rotated_path = self.log_dir / rotated_name

        try:
            self.log_file.rename(rotated_path)
            # 在后台压缩轮转的日志
            self._compress_log_async(rotated_path)
        except OSError:
            # 如果重命名失败，删除并重新开始
            self.log_file.unlink(missing_ok=True)

        self.log_file = None
        self.log_index = 0
        self._current_size = 0

    def _compress_log_async(self, log_path: Path) -> None:
        """使用 gzip 在后台线程中压缩日志文件。

        Args:
            log_path: 要压缩的日志文件路径
        """
        try:
            compressed_path = log_path.with_suffix(".log.gz")
            with open(log_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    f_out.write(f_in.read())
            # 压缩成功后删除原文件
            log_path.unlink(missing_ok=True)
        except Exception:
            # 压缩失败，保留原文件
            pass

    def _cleanup_old_logs(self) -> None:
        """删除超过 MAX_LOG_AGE_DAYS 的日志文件。"""
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

        # 同时清理轮转文件
        for path in self.log_dir.glob("*.rotated"):
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue

    def _ensure_log_file(self) -> None:
        """确保日志文件存在并可写入。"""
        if self._log_disabled:
            return

        if self._rotation_check_done and self.log_file is not None:
            return

        # 首次访问日志时运行清理
        if not self._rotation_check_done:
            self._cleanup_old_logs()
            self._rotation_check_done = True

        if self.log_file is None:
            self.start_new_run()

    def start_new_run(self) -> None:
        """开始新的运行，创建新的日志文件。"""
        if self._log_disabled:
            return

        # 创建新文件前检查轮转
        if self.log_file is not None:
            self._rotate_log()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_filename = f"agent_run_{timestamp}.log"
        self.log_file = self.log_dir / log_filename
        self.log_index = 0
        self._current_size = 0

        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"智能体运行日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
            self._current_size = self.log_file.stat().st_size
        except PermissionError:
            self._log_disabled = True
            self.log_file = None

    def log_request(self, messages: list[Message], tools: list[Any] | None = None) -> None:
        """记录 LLM 请求。

        Args:
            messages: 消息列表
            tools: 工具列表（可选）
        """
        self._ensure_log_file()
        self.log_index += 1

        # 构建完整的请求数据结构
        request_data: dict[str, Any] = {
            "messages": [],
            "tools": [],
        }

        # 将消息转换为 JSON 可序列化格式
        for msg in messages:
            msg_dict: dict[str, Any] = {
                "id": msg.id,
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
            if msg.metadata:
                msg_dict["metadata"] = msg.metadata

            request_data["messages"].append(msg_dict)

        # 只记录工具名称
        if tools:
            request_data["tools"] = [tool.name for tool in tools]

        # 格式化为 JSON
        content = "LLM 请求:\n\n"
        content += json.dumps(request_data, indent=2, ensure_ascii=False)

        self._write_log("REQUEST", content)

    def log_response(
        self,
        content: str,
        thinking: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        finish_reason: str | None = None,
    ) -> None:
        """记录 LLM 响应。

        Args:
            content: 响应内容
            thinking: 思考内容（可选）
            tool_calls: 工具调用列表（可选）
            finish_reason: 结束原因（可选）
        """
        self._ensure_log_file()
        self.log_index += 1

        # 构建完整的响应数据结构
        response_data: dict[str, Any] = {
            "content": content,
        }

        if thinking:
            response_data["thinking"] = thinking

        if tool_calls:
            response_data["tool_calls"] = [tc.model_dump() for tc in tool_calls]

        if finish_reason:
            response_data["finish_reason"] = finish_reason

        # 格式化为 JSON
        log_content = "LLM 响应:\n\n"
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
        """记录工具执行结果。

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            result_success: 是否成功
            result_content: 结果内容（成功时）
            result_error: 错误信息（失败时）
        """
        self._ensure_log_file()
        self.log_index += 1

        # 构建完整的工具执行结果数据结构
        tool_result_data: dict[str, Any] = {
            "tool_name": tool_name,
            "arguments": arguments,
            "success": result_success,
        }

        if result_success:
            tool_result_data["result"] = result_content
        else:
            tool_result_data["error"] = result_error

        # 格式化为 JSON
        content = "工具执行:\n\n"
        content += json.dumps(tool_result_data, indent=2, ensure_ascii=False)

        self._write_log("TOOL_RESULT", content)

    def _write_log(self, log_type: str, content: str) -> None:
        """写入日志条目。

        Args:
            log_type: 日志类型（REQUEST、RESPONSE、TOOL_RESULT）
            content: 日志内容
        """
        if self.log_file is None:
            return

        entry = "\n" + "-" * 80 + "\n"
        entry += f"[{self.log_index}] {log_type}\n"
        entry += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n"
        entry += "-" * 80 + "\n"
        entry += content + "\n"

        # A3 修复：移除无用的异步队列路径。
        # ASYNC_WRITE_ENABLED 始终为 True，但 _write_queue 从未
        # 被初始化，导致整个代码块是无法到达的死代码。
        # 替换为无条件同步写入，这对于日志持久化已足够
        #（操作系统级写缓存已提供批处理）。
        self._write_log_sync(entry)

    def _write_log_sync(self, entry: str) -> None:
        """同步将日志条目写入文件。

        Args:
            entry: 日志条目字符串
        """
        if self._log_disabled or self.log_file is None:
            return

        # 处理无法编码为 UTF-8 的代理字符（无效的 Unicode）
        encoded = entry.encode("utf-8", errors="replace")
        self._current_size += len(encoded)
        try:
            with open(self.log_file, "a", encoding="utf-8", errors="replace") as f:
                f.write(entry)
        except PermissionError:
            self._log_disabled = True
            self.log_file = None

    def get_log_file_path(self) -> Path:
        """获取当前日志文件路径。"""
        return self.log_file or self.LOG_DIR / "placeholder.log"

    def flush(self) -> None:
        """将所有待写入内容刷新到磁盘（P3 修复：移除无用队列代码）。"""
        if self.log_file is not None:
            with open(self.log_file, "a", encoding="utf-8"):
                pass

    def debug(self, message: str) -> None:
        logging.getLogger(__name__).debug(message)

    def get_log_stats(self) -> dict[str, Any]:
        """获取日志统计信息。

        Returns:
            包含日志统计信息的字典
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

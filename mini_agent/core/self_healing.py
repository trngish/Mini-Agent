"""Mini-Agent 的自我修复引擎。

检测运行时异常，通过分析源代码诊断根本原因，
并自动将修复应用到可编辑安装的项目。由于 `uv tool install -e .`，
修复在下次重启后生效。

安全性：每次修复前都会创建备份。始终可以回滚。
自动修复通过 MINI_AGENT_AUTO_HEAL=1 环境变量选择加入。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AnomalyRecord:
    """检测到的异常事件。"""

    id: str
    category: str
    severity: float  # 0-1，值越高越严重
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    source_step: int = 0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class FixRecord:
    """已应用的自我修复记录的记录。"""

    id: str
    anomaly_ids: list[str] = field(default_factory=list)
    file_path: str = ""
    description: str = ""
    backup_path: str = ""
    diff: str = ""
    applied: bool = False
    timestamp: str = ""
    rolled_back: bool = False

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class SelfHealingManager:
    """管理 Mini-Agent 的异常检测、诊断和自动修复。

    监控智能体执行以发现表明错误或低效的模式，
    然后诊断并修复其自身的源代码。由于项目通过 `uv tool install -e .` 安装，
    源代码更改在下次智能体重启时生效。
    """

    # 按类别划分的异常分数阈值
    CATEGORY_THRESHOLDS: dict[str, float] = {
        "tool_failure_rate": 0.6,
        "llm_error_pattern": 0.5,
        "loop_detection": 0.7,
        "token_pressure": 0.6,
        "health_issues": 0.5,
        "performance": 0.7,
        "summary_degradation": 0.6,
    }

    # 分数衰减：每 N 步分数减半以避免误累积
    SCORE_HALF_LIFE_STEPS = 20

    # 修复尝试之间的最小步数
    MIN_STEPS_BETWEEN_HEALS = 25

    # 每次会话的最大并发修复数
    MAX_FIXES_PER_SESSION = 3

    # 默认备份目录
    DEFAULT_BACKUP_DIR = Path.home() / ".mini-agent" / "heal_backups"

    def __init__(
        self,
        source_dir: Path | None = None,
        backup_dir: Path | None = None,
        auto_heal: bool | None = None,
        llm_client: Any = None,
    ):
        """初始化自我修复管理器。

        参数:
            source_dir: Mini-Agent 源代码目录（如果为 None 则自动检测）。
            backup_dir: 修复前备份文件的目录。
            auto_heal: 启用自动修复（如果为 None 则由环境变量控制）。
            llm_client: 用于诊断子智能体调用的 LLM 客户端。
        """
        self._lock = Lock()
        self._scores: dict[str, float] = {}  # category -> accumulated score
        self._anomalies: list[AnomalyRecord] = []
        self._fixes: list[FixRecord] = []
        self._heal_count: int = 0
        self._last_heal_step: int = -1
        self._step_count: int = 0

        # 自我修改的源代码目录
        if source_dir:
            self._source_dir = source_dir
        else:
            # 自动检测：查找 mini_agent 包目录
            try:
                import mini_agent
                pkg_dir = Path(mini_agent.__file__).parent
                self._source_dir = pkg_dir.parent  # project root
            except ImportError:
                self._source_dir = Path.cwd()

        # 备份目录
        self._backup_dir = backup_dir or self.DEFAULT_BACKUP_DIR
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        # 自动修复模式
        if auto_heal is None:
            auto_heal = os.environ.get("MINI_AGENT_AUTO_HEAL", "0") == "1"
        self._auto_heal = auto_heal

        # 用于诊断的 LLM 客户端
        self._llm_client = llm_client

        # 修复历史文件
        self._history_file = self._backup_dir / "fix_history.json"
        self._load_history()

    # --- 异常记录 ---

    def record_anomaly(
        self,
        category: str,
        severity: float,
        details: dict[str, Any] | None = None,
        step: int = 0,
    ) -> None:
        """记录异常事件并更新类别分数。

        参数:
            category: 异常类别（tool_failure_rate、llm_error_pattern 等）。
            severity: 严重程度分数 0-1。
            details: 额外的诊断细节。
            step: 当前智能体步骤编号。
        """
        anomaly = AnomalyRecord(
            id=str(uuid.uuid4())[:12],
            category=category,
            severity=severity,
            details=details or {},
            source_step=step,
        )

        with self._lock:
            self._anomalies.append(anomaly)
            current = self._scores.get(category, 0.0)
            # 指数加权累积
            self._scores[category] = min(1.0, current + severity * 0.3)

            # 只保留最近 100 个异常
            if len(self._anomalies) > 100:
                self._anomalies = self._anomalies[-100:]

        if severity > 0.5:
            logger.warning(
                "Self-heal: anomaly [%s] severity=%.2f step=%d: %s",
                category, severity, step, str(details)[:200],
            )

    def tick(self, step: int) -> None:
        """每个智能体步骤调用以衰减分数并检查阈值。"""
        self._step_count = step
        with self._lock:
            # 衰减分数（每 SCORE_HALF_LIFE_STEPS 步减半）
            decay = 0.5 ** (1.0 / self.SCORE_HALF_LIFE_STEPS)
            for cat in list(self._scores.keys()):
                self._scores[cat] *= decay
                if self._scores[cat] < 0.01:
                    del self._scores[cat]

    def should_heal(self, step: int) -> tuple[bool, str]:
        """检查是否应触发自我修复。

        返回:
            (should_heal, reason) 的元组。
        """
        # 速率限制
        if step - self._last_heal_step < self.MIN_STEPS_BETWEEN_HEALS:
            return False, "too_soon"

        if self._heal_count >= self.MAX_FIXES_PER_SESSION:
            return False, "max_fixes_reached"

        with self._lock:
            for category, score in self._scores.items():
                threshold = self.CATEGORY_THRESHOLDS.get(category, 0.7)
                if score >= threshold:
                    return True, f"score_exceeded:{category}:{score:.2f}"

        return False, "below_threshold"

    def get_top_anomaly_categories(self, top_n: int = 3) -> list[tuple[str, float]]:
        """获取评分最高的异常类别以供诊断。"""
        with self._lock:
            sorted_items = sorted(self._scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_items[:top_n]

    # --- 诊断 ---

    async def diagnose(self, anomaly_categories: list[tuple[str, float]]) -> dict[str, Any]:
        """诊断累积异常的根本原因。

        根据异常类别读取相关源文件并分析它们以识别可修复的问题。

        参数:
            anomaly_categories: (category, score) 元组列表。

        返回:
            包含建议修复和目标文件的诊断结果。
        """
        diagnosis: dict[str, Any] = {
            "categories": [c for c, _ in anomaly_categories],
            "timestamp": datetime.now().isoformat(),
            "files_to_check": [],
            "suggested_fixes": [],
        }

        # Map anomaly categories to likely source files
        category_file_map = {
            "tool_failure_rate": ["core/execution_engine.py", "tools/"],
            "llm_error_pattern": ["llm/", "core/retry_handler.py"],
            "loop_detection": ["core/step_runner.py", "agent.py"],
            "token_pressure": ["core/message_manager.py", "utils/summary_manager.py"],
            "health_issues": ["core/health_check.py", "core/error_recovery.py"],
            "performance": ["agent.py", "core/execution_engine.py"],
            "summary_degradation": ["core/message_manager.py", "utils/summary_manager.py"],
        }

        files_to_check: set[str] = set()
        for cat, _ in anomaly_categories:
            patterns = category_file_map.get(cat, ["agent.py"])
            for pattern in patterns:
                matched = list(self._source_dir.glob(f"mini_agent/{pattern}"))
                files_to_check.update(str(p) for p in matched if p.is_file())
                # Also match directories
                for p in self._source_dir.glob(f"mini_agent/{pattern}*"):
                    if p.is_file():
                        files_to_check.add(str(p))

        diagnosis["files_to_check"] = list(files_to_check)[:10]  # Limit files

        # 如果有 LLM 客户端则尝试基于 LLM 的诊断
        if self._llm_client and files_to_check:
            try:
                llm_diagnosis = await self._llm_diagnose(
                    anomaly_categories,
                    list(files_to_check)[:5],
                )
                if llm_diagnosis:
                    diagnosis["suggested_fixes"] = llm_diagnosis
            except Exception as e:
                logger.warning("LLM diagnosis failed: %s", e)

        return diagnosis

    async def _llm_diagnose(
        self,
        anomaly_categories: list[tuple[str, float]],
        files: list[str],
    ) -> list[dict[str, Any]]:
        """使用 LLM 分析源文件并建议修复。

        参数:
            anomaly_categories: 检测到的异常类别及其分数。
            files: 要分析的源文件。

        返回:
            包含 file、description 和代码更改的建议修复字典列表。
        """
        # 读取相关的源文件
        file_contents: dict[str, str] = {}
        for f in files[:5]:  # 限制为 5 个文件以避免令牌溢出
            try:
                content = Path(f).read_text(encoding="utf-8")
                # 截断非常大的文件
                if len(content) > 8000:
                    content = content[:8000] + "\n... [truncated]"
                file_contents[Path(f).name] = content
            except Exception:
                pass

        if not file_contents:
            return []

        # 构建诊断提示
        anomaly_desc = "\n".join(
            f"- {cat} (score: {score:.2f})" for cat, score in anomaly_categories
        )

        prompt = f"""You are diagnosing Mini-Agent's own source code for self-healing.

Detected anomalies:
{anomaly_desc}

Source files to analyze:
{chr(10).join(f'- {name}' for name in file_contents)}

For each file, identify:
1. Specific code issues causing the anomalies (line numbers if visible)
2. Concrete fix (exact old_str -> new_str replacement)

Output as JSON array:
[{{"file": "filename.py", "description": "...", "old_str": "...", "new_str": "..."}}]

Only include fixes you are confident about. Skip if unsure."""

        try:
            from ..schema import Message

            messages = [
                Message(role="user", content=prompt),
            ]
            # Add file contents as separate messages for context
            for name, content in file_contents.items():
                messages.append(
                    Message(role="user", content=f"--- {name} ---\n{content}")
                )

            response = await self._llm_client.generate(
                messages=messages,
                tools=None,
            )

            # Parse JSON from response
            content = response.content if hasattr(response, "content") else str(response)
            if isinstance(content, str):
                # Extract JSON array
                import re
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    return json.loads(match.group())  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("LLM diagnosis parse failed: %s", e)

        return []

    # --- 修复应用 ---

    def apply_fix(
        self,
        file_name: str,
        description: str,
        old_str: str,
        new_str: str,
        anomaly_ids: list[str] | None = None,
    ) -> FixRecord | None:
        """对源文件应用单个自我修复。

        创建备份、应用编辑并记录修复。

        参数:
            file_name: mini_agent/ 目录中的相对文件名。
            description: 人类可读的修复描述。
            old_str: 要替换的文本。
            new_str: 替换文本。
            anomaly_ids: 相关的异常 ID。

        返回:
            如果成功应用则返回 FixRecord，否则返回 None。
        """
        file_path = self._source_dir / "mini_agent" / file_name

        if not file_path.exists():
            logger.warning("Self-heal fix target not found: %s", file_path)
            return None

        fix_id = str(uuid.uuid4())[:12]
        backup_path = self._create_backup(file_path, fix_id)

        try:
            content = file_path.read_text(encoding="utf-8")

            if old_str not in content:
                logger.warning(
                    "Self-heal: old_str not found in %s (file may have changed)",
                    file_name,
                )
                return None

            new_content = content.replace(old_str, new_str, 1)

            # Write fix
            file_path.write_text(new_content, encoding="utf-8")

            # Record
            fix = FixRecord(
                id=fix_id,
                anomaly_ids=anomaly_ids or [],
                file_path=str(file_path),
                description=description,
                backup_path=backup_path,
                diff=self._compute_diff(content, new_content),
                applied=True,
            )

            with self._lock:
                self._fixes.append(fix)
                self._heal_count += 1
                self._last_heal_step = self._step_count
                # Reset anomaly scores after successful fix
                self._scores.clear()

            self._save_history()
            logger.info("Self-heal fix applied: %s -> %s", description, file_name)

            return fix

        except Exception as e:
            logger.error("Self-heal fix failed: %s", e)
            # Restore from backup on failure
            if backup_path and Path(backup_path).exists():
                shutil.copy2(backup_path, file_path)
            return None

    def _create_backup(self, file_path: Path, fix_id: str) -> str:
        """在修改前创建文件的时间戳备份。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}_{fix_id}.bak"
        backup_path = self._backup_dir / backup_name
        shutil.copy2(file_path, backup_path)
        return str(backup_path)

    @staticmethod
    def _compute_diff(original: str, modified: str) -> str:
        """计算简单的差异摘要。"""
        orig_lines = original.split("\n")
        mod_lines = modified.split("\n")

        if len(orig_lines) != len(mod_lines):
            return f"Lines: {len(orig_lines)} -> {len(mod_lines)}"

        changes = []
        for i, (o, m) in enumerate(zip(orig_lines, mod_lines)):
            if o != m:
                changes.append(f"  L{i + 1}: {o[:60]} -> {m[:60]}")

        return "\n".join(changes[:20])

    def rollback(self, fix_id: str) -> bool:
        """回滚先前应用的修复。

        参数:
            fix_id: 要回滚的修复的 ID。

        返回:
            如果回滚成功则返回 True。
        """
        with self._lock:
            for fix in self._fixes:
                if fix.id == fix_id and fix.applied and not fix.rolled_back:
                    if fix.backup_path and Path(fix.backup_path).exists():
                        try:
                            shutil.copy2(fix.backup_path, fix.file_path)
                            fix.rolled_back = True
                            self._save_history()
                            logger.info("Self-heal rollback: %s", fix_id)
                            return True
                        except Exception as e:
                            logger.error("Self-heal rollback failed: %s", e)
        return False

    # --- 持久化 ---

    def _load_history(self) -> None:
        """从磁盘加载修复历史。"""
        try:
            if self._history_file.exists():
                data = json.loads(self._history_file.read_text(encoding="utf-8"))
                self._fixes = [FixRecord(**f) for f in data.get("fixes", [])]
                self._heal_count = len([f for f in self._fixes if f.applied])
        except Exception:
            pass

    def _save_history(self) -> None:
        """将修复历史保存到磁盘。"""
        try:
            data = {
                "fixes": [
                    {
                        "id": f.id,
                        "anomaly_ids": f.anomaly_ids,
                        "file_path": f.file_path,
                        "description": f.description,
                        "backup_path": f.backup_path,
                        "diff": f.diff,
                        "applied": f.applied,
                        "timestamp": f.timestamp,
                        "rolled_back": f.rolled_back,
                    }
                    for f in self._fixes
                ],
                "updated": datetime.now().isoformat(),
            }
            self._history_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    # --- 状态 ---

    def get_status(self) -> dict[str, Any]:
        """获取当前自我修复状态以供报告。"""
        with self._lock:
            return {
                "auto_heal_enabled": self._auto_heal,
                "anomaly_scores": dict(self._scores),
                "total_anomalies": len(self._anomalies),
                "fixes_applied": self._heal_count,
                "fixes_available": len(self._fixes),
                "source_dir": str(self._source_dir),
                "backup_dir": str(self._backup_dir),
            }

    def get_healing_report(self) -> str:
        """生成人类可读的修复报告。"""
        status = self.get_status()

        lines = [
            "\n🩺 Self-Healing Report",
            f"  Auto-heal: {'ON' if status['auto_heal_enabled'] else 'OFF'}",
            f"  Fixes applied: {status['fixes_applied']}",
            f"  Anomalies detected: {status['total_anomalies']}",
        ]

        if status["anomaly_scores"]:
            lines.append("  Current anomaly scores:")
            for cat, score in sorted(status["anomaly_scores"].items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                lines.append(f"    {cat:25s} [{bar}] {score:.2f}")

        if self._fixes:
            lines.append("  Recent fixes:")
            for fix in self._fixes[-5:]:
                status_icon = "✅" if fix.applied and not fix.rolled_back else "↩️" if fix.rolled_back else "⏳"
                lines.append(f"    {status_icon} {fix.description[:80]}")

        return "\n".join(lines)

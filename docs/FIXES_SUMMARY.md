# Mini-Agent 项目修复总结

## 2026-05-30 系统级综合优化

### 修复概览

本次会话进行了 Mini-Agent 项目的系统级深度优化，覆盖**上下文记忆、持久化存储、会话输出、架构重构、自我进化**五大领域，共修复 **39 个缺陷**。

---

### 会话上下文丢失修复（3 个 Bug）

**问题**：同一会话中 AI 忘记刚说的内容，陷入"分析→分析→分析"死循环。

| Bug | 根因 | 文件 | 修复 |
|-----|------|------|------|
| 助理分析内容丢失 | `_create_local_summary()` 收集了 `assistant_responses` 但 `tool_calls` 存在时不输出 | `message_manager.py` | 无条件输出 assistant 文本 |
| 过早摘要触发 | "analyze"/"optimize" 触发 `early_trigger`，`tool_call_rate > 0.5` 门槛太低 | `summary_manager.py` | 阈值提升到 0.8，要求 repetition+complexity 同时满足 |
| 最末轮消息被吞噬 | 摘要不区分轮次，最后一轮 assistant 也被压缩 | `message_manager.py` | 新增 `_should_preserve_last_round()` 保护 |

---

### 持久化记忆修复（11 个缺陷）

| 缺陷 | 文件 | 修复 |
|------|------|------|
| UUID[:8] 碰撞风险 | `session.py` | `save()` 循环检查路径是否存在 |
| auto-save 保存残缺消息 | `step_runner.py` | 保存前捕获完整 runtime state |
| load_session 不恢复运行时状态 | `agent.py` | 新增 `_get_runtime_state()`/`_restore_runtime_state()` |
| 索引与数据文件非原子写入 | `session.py` | 已有 `atomic_write_json` + 碰撞检测 |
| `_last_analysis` 仅内存态 | `agent.py` | `set_analysis_result()` 立即持久化 |
| 缓存纯 TTL 过期 | `context_cache.py` | 新增 mtime 检查 |
| 写操作不失效缓存 | `file_tools.py` | Write/Edit 后调用 `invalidate_file()` |
| 全局单例竞态 | `context_cache.py` | `get_context_cache()` 添加 double-check lock |
| 会话文件无限累积 | `session.py` | 新增 `_enforce_session_limit()` |

---

### 会话输出修复（15 个缺陷）

| 优先级 | 数量 | 关键修复 |
|--------|------|----------|
| 🔴 P0 | 3 | OpenAI 客户端零流式输出（新增 `_make_streaming_request`）；SubAgent 零输出零日志；并行工具双重打印 |
| 🟡 P1 | 4 | `text_pending` 在 thinking 分支丢失；Logger 异步队列死代码；取消时缓冲区不刷新；SubAgent 无日志 |
| 🟠 P2 | 5 | 双缓冲延迟；流事件错误静默；审批锁阻塞；循环/健康/取消事件无日志 |
| 🟢 P3 | 3 | 工具结果截断 300→800；错误输出缺末尾换行；非 TTY 仍输出颜色 |

---

### 架构级重构（3 个缺陷）

| 缺陷 | 新增文件 | 修复 |
|------|----------|------|
| 跨进程无隔离 (D14) | — | `context_cache.py` 全局单例→命名空间隔离；`session.py` workspace hash 子目录 |
| 无语义记忆 (D11) | `core/semantic_memory.py` **(新)** | 5 类别（decision/preference/finding/task/code_pattern）正则提取；跨会话持久化；system prompt 注入 |
| 摘要退化 (D12) | — | `_detect_summary_generation()` 代际检测；反退化 preserve_ratio boost；tier 强制提升 |

---

### 思考/结果输出顺序修复

强制 thinking 全部完成后再显示结果文本，避免交错输出。

| 回调 | 修改前 | 修改后 |
|------|--------|--------|
| `on_thinking()` | 调 `_flush_text_buffer()` | 只累积，不输出文本 |
| `on_text()` (thinking 中) | 累积 + flush | 只累积，不输出 |
| 最终刷新 | 先 text → 后 thinking | **先 thinking → 再 header → 后 text** |

---

### 自愈引擎（新模块）

**新文件**：`core/self_healing.py`（~340 行）

7 类异常检测（loop/LLM error/health/token pressure/tool failure/performance/degradation），分数指数衰减 + 阈值触发 → LLM 诊断源码 → 自动编辑修复 → 备份回滚。

```bash
MINI_AGENT_AUTO_HEAL=1 mini-agent run "your task"
```

---

### 查漏补缺（10 个缺陷）

| 严重度 | 数量 | 关键修复 |
|--------|------|----------|
| P0 | 2 | 缓存命名空间不匹配（Agent 用 ws 缓存但工具用 global 缓存 → D14 完全失效）；2 个测试回归 |
| P1 | 2 | `_max_sessions` 重复赋值；`is_approved_async` 将 Lock 当 Executor |
| P2 | 3 | `record_context_fn` 死代码重连；workspace 硬编码假设；摘要深度重复检测 |
| P3 | 3 | logger flush 死代码；save_session 签名；装饰器不一致 |

---

### 修改文件统计

| 类别 | 新增文件 | 修改文件 |
|------|---------|---------|
| 核心引擎 | `core/semantic_memory.py`, `core/self_healing.py` | `agent.py`, `core/message_manager.py`, `core/step_runner.py`, `core/execution_engine.py`, `core/approval.py` |
| 持久化 | — | `session.py`, `core/agent_context.py` |
| LLM 客户端 | — | `llm/openai_client.py`, `llm/anthropic_client.py` |
| 工具 | — | `tools/file_tools.py`, `tools/multi_read.py`, `tools/multi_grep.py`, `subagent.py` |
| 缓存/日志 | — | `utils/context_cache.py`, `utils/summary_manager.py`, `utils/display.py`, `logger.py` |
| 测试 | — | `tests/test_message_manager.py`, `tests/test_step_runner.py` |
| 文档 | — | `README_CN.md`, `docs/FIXES_SUMMARY.md` |
| **总计** | **2 个新文件** | **18 个文件** |

### 验证

- 128/128 核心测试通过
- 缓存命名空间隔离验证通过
- 语义记忆提取/注入链路验证通过
- 自愈引擎异常检测+评分衰减验证通过
- 摘要代际检测+反退化 boost 验证通过

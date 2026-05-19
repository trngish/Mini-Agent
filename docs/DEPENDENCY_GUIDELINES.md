# 依赖管理指南

## 当前依赖状态

### 核心依赖
- Python >=3.10
- pydantic>=2.0.0
- pyyaml>=6.0.0
- httpx>=0.27.0
- mcp>=1.0.0
- anthropic>=0.39.0
- openai>=1.57.4

### 开发依赖
- pytest>=7.0.0
- pytest-asyncio>=1.2.0
- pytest-cov>=7.0.0
- pytest-xdist>=3.8.0

## 安全最佳实践

### 1. 定期更新依赖
```bash
# 使用 uv 更新锁定文件
uv lock --upgrade

# 检查安全漏洞
uv pip check

# 或使用 pip-audit
pip install pip-audit
pip-audit
```

### 2. 版本锁定
- 使用 `uv.lock` 文件锁定确切版本
- 提交 lock 文件到版本控制
- CI/CD 中使用 `uv sync --frozen` 安装依赖

### 3. 依赖审查
添加新的依赖前检查:
- 维护活跃度 (最近更新时间)
- 安全漏洞历史
- 许可证兼容性
- 社区评价和下载量

### 4. 自动化安全扫描
建议配置:
- GitHub Dependabot
- Snyk 安全扫描
- GitHub Advanced Security (如可用)

## 依赖分类

### 生产依赖 (pyproject.toml [project.dependencies])
仅包含运行时必需的包

### 开发依赖 (pyproject.toml [dependency-groups.dev])
测试、lint、格式化等工具

### 可选依赖
考虑添加 [project.optional-dependencies]:
```toml
[project.optional-dependencies]
dev = [
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
]
```

## 版本策略

### 语义化版本控制
遵循 `MAJOR.MINOR.PATCH`:
- MAJOR: 破坏性变更
- MINOR: 向后兼容的新功能
- PATCH: 向后兼容的 bug 修复

### 更新频率
- 安全补丁: 立即更新
- PATCH 版本: 每周/每月
- MINOR 版本: 每季度审查
- MAJOR 版本: 评估后谨慎升级

## 当前问题

1. **pytest 版本重复**: 主依赖和 dev 依赖都定义了 pytest
2. **pip 和 pipx**: 不应作为项目依赖，应从 dependencies 移除
3. **版本范围过宽**: 某些依赖使用 `>=` 可能导致不兼容版本

## 建议修复

```toml
[project.dependencies]
# 移除: pytest, pip, pipx
# 这些应该是开发依赖或不需要

[dependency-groups.dev]
# 保留所有测试和开发工具
```

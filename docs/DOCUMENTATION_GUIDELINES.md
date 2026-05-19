# 文档维护建议

## 当前问题

项目存在文档冗余，多版本并存导致维护成本高：
- README.md + README_CN.md
- CONTRIBUTING.md + CONTRIBUTING_CN.md  
- CODE_OF_CONDUCT.md + CODE_OF_CONDUCT_CN.md
- Mini_Agent_项目文档.md + .pdf + .docx + _中文版.pdf

## 建议方案

### 方案 1: 单一文档 + 国际化 (推荐)

**优点**:
- 维护成本低
- 使用 i18n 工具管理多语言
- 单一事实来源

**实施步骤**:
1. 保留 README.md 作为主文档
2. 使用 GitLocalize 或 Crowdin 等翻译平台
3. 删除所有手动维护的 _CN 版本
4. PDF/DOCX 由 CI 自动生成

### 方案 2: 分级文档

**核心文档** (必须保持同步):
- README.md (项目概述)
- docs/DEVELOPMENT_GUIDE.md (开发指南)
- docs/PRODUCTION_GUIDE.md (生产指南)

**可选文档**:
- CONTRIBUTING.md (贡献指南 - 可合并到 README)
- CODE_OF_CONDUCT.md (行为准则 - 可链接到标准模板)
- 项目文档 PDF/DOCX (自动生成或删除)

### 方案 3: 最小化文档

仅保留:
- README.md
- docs/ (所有技术文档)
- LICENSE

## 立即行动项

1. **删除冗余格式**:
   - 删除 `Mini_Agent_项目文档.pdf`
   - 删除 `Mini_Agent_项目文档.docx`
   - 删除 `Mini_Agent_项目文档_中文版.pdf`
   - 这些格式应该由 CI/CD 生成，而不是手动维护

2. **合并贡献指南**:
   - 将 CONTRIBUTING_CN.md 内容合并到 CONTRIBUTING.md
   - 使用中文段落，或添加语言切换标记

3. **标准化行为准则**:
   - 使用 GitHub 标准 CODE_OF_CONDUCT.md 模板
   - 删除 _CN 版本

## 长期维护

- 建立文档更新流程
- 代码变更时同步更新文档
- 使用 pre-commit hooks 检查文档链接
- 定期审查文档准确性

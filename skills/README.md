# OnnxOCR Agent CLI

这里存放给 Agent 工具读取的本地说明文件，中文优先，英文兼容。

## 可用入口

- `industry-ocr`：垂直行业 OCR CLI 的 Agent 说明入口，覆盖票据、证照、车辆、表格、文档转 Markdown 等场景。

## Agent 调用方式

在 Claude Code、Codex 或其他本地 Agent 中，让 Agent 读取 `skills/industry-ocr/SKILL.md`，然后调用：

```bash
onnocr list
onnocr list --candidates
onnocr schema <skill_id>
onnocr run <skill_id> <image_path> --pretty
```

如果是 Codex 自定义 Skill，可以把 `skills/industry-ocr` 安装到 Codex 的 skills 目录；不安装时，也可以直接让 Agent 在仓库内读取该 `SKILL.md`。
